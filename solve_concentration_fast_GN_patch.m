function [cH, cE] = solve_concentration_fast_GN_patch(OD_in, gH, gE, gMat, den, opts)
%SOLVE_CONCENTRATION_FAST_GN (分块优化版)
% 向量化 Gauss-Newton / LM 求解全图 (cH,cE)，支持超大分辨率图像分块处理。

    if nargin < 6 || isempty(opts), opts = struct(); end
    if ~isfield(opts,'maxIter'), opts.maxIter = 10; end
    if ~isfield(opts,'lambda0'), opts.lambda0 = 1e-2; end
    if ~isfield(opts,'lambdaDecay'), opts.lambdaDecay = 0.5; end
    if ~isfield(opts,'lambdaMin'), opts.lambdaMin = 1e-8; end
    if ~isfield(opts,'cMax'), opts.cMax = Inf; end
    if ~isfield(opts,'epsRGB'), opts.epsRGB = 1e-8; end
    if ~isfield(opts,'tinyToZero'), opts.tinyToZero = 0; end 
    
    % 新增：分块大小（默认50万像素，可根据你的电脑内存自行调大或调小）
    if ~isfield(opts,'chunkSize'), opts.chunkSize = 500000; end 

    % -----------------------------
    % 1) 处理输入形状
    % -----------------------------
    sz = size(OD_in);
    if ndims(OD_in) == 3
        m = sz(1); n = sz(2);
        OD = reshape(OD_in, [], 3);  % Nx3
        isImage = true;
    else
        OD = OD_in;                 % Nx3
        isImage = false;
        m = []; n = [];
    end
    if size(OD,2) ~= 3
        error('OD_in must be Nx3 or mxnx3.');
    end
    N = size(OD,1);

    % -----------------------------
    % 2) 固定 forward 模型参数
    % -----------------------------
    W = gMat';
    if size(W,1) ~= 3
        error('gMat'' should be 3xK so that (gMat'' * T) gives 3xN.');
    end
    Spec = [gH(:), gE(:)];
    K = size(Spec,1);
    if size(W,2) ~= K
        error('Size mismatch: size(gMat'',2) must equal length(gH/gE).');
    end
    den = den(:);
    
    % 提前算出 3x2 的近似 A 矩阵 (只算一次)
    OD0 = forwardOD_fromC([0;0], Spec, W, den, opts.epsRGB);
    ODH = forwardOD_fromC([1;0], Spec, W, den, opts.epsRGB);
    ODE = forwardOD_fromC([0;1], Spec, W, den, opts.epsRGB);
    A = [ODH-OD0, ODE-OD0]; 
    A_pinv = pinv(A);       

    % 初始化输出容器
    cH_full = zeros(N, 1);
    cE_full = zeros(N, 1);

    % -----------------------------
    % 3) 分块主循环 (防内存溢出核心逻辑)
    % -----------------------------
    numChunks = ceil(N / opts.chunkSize);
    
    for c_idx = 1:numChunks
        % 计算当前块的索引范围
        startIdx = (c_idx - 1) * opts.chunkSize + 1;
        endIdx   = min(c_idx * opts.chunkSize, N);
        
        % 提取当前块并转置为 3xN_chunk
        OD3_chunk = OD(startIdx:endIdx, :)'; 
        
        % 初始化浓度 C
        C_chunk = A_pinv * (OD3_chunk - OD0);
        C_chunk = max(C_chunk, 0);
        if isfinite(opts.cMax)
            C_chunk = min(C_chunk, opts.cMax);
        end
        
        % 当前块的 LM / Gauss-Newton 迭代
        lambda = opts.lambda0;
        for it = 1:opts.maxIter
            [OD_hat, RGB, T] = forwardOD_batch(C_chunk, Spec, W, den, opts.epsRGB);
            r = OD_hat - OD3_chunk;
            
            TS1 = T .* Spec(:,1);
            TS2 = T .* Spec(:,2);
            J1 = (W * TS1) ./ (den .* RGB);
            J2 = (W * TS2) ./ (den .* RGB);
            
            a11 = sum(J1 .* J1, 1) + lambda;
            a22 = sum(J2 .* J2, 1) + lambda;
            a12 = sum(J1 .* J2, 1);
            b1  = sum(J1 .* r, 1);
            b2  = sum(J2 .* r, 1);
            
            detM = a11 .* a22 - a12 .* a12 + 1e-12;
            d1 = -( a22 .* b1 - a12 .* b2 ) ./ detM;
            d2 = -( -a12 .* b1 + a11 .* b2 ) ./ detM;
            
            C_chunk(1,:) = C_chunk(1,:) + d1;
            C_chunk(2,:) = C_chunk(2,:) + d2;
            
            C_chunk = max(C_chunk, 0);
            if isfinite(opts.cMax)
                C_chunk = min(C_chunk, opts.cMax);
            end
            if opts.tinyToZero > 0
                C_chunk(C_chunk < opts.tinyToZero) = 0;
            end
            
            lambda = max(lambda * opts.lambdaDecay, opts.lambdaMin);
        end
        
        % 将计算结果存入全局容器
        cH_full(startIdx:endIdx) = C_chunk(1,:)';
        cE_full(startIdx:endIdx) = C_chunk(2,:)';
    end

    % -----------------------------
    % 4) 输出形状
    % -----------------------------
    if isImage
        cH = reshape(cH_full, m, n);
        cE = reshape(cE_full, m, n);
    else
        cH = cH_full;
        cE = cE_full;
    end
end

% ========================================================================
% Local helper functions 保持完全不变
% ========================================================================
function [OD_hat, RGB, T] = forwardOD_batch(C, Spec, W, den, epsRGB)
    Exponent = -(Spec * C);
    T = 10 .^ Exponent;
    RGB = (W * T) ./ den;
    RGB = max(RGB, epsRGB);
    OD_hat = -log10(RGB);
end

function OD = forwardOD_fromC(c, Spec, W, den, epsRGB)
    Exponent = -(Spec * c);
    T = 10 .^ Exponent;
    RGB = (W * T) ./ den;
    RGB = max(RGB, epsRGB);
    OD = -log10(RGB);
end