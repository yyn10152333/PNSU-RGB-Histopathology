function he_spectral_decomposition( refImagePath,inputDir, outRoot, C_list, P_list, ~, varargin)
% 批量 HE 分离 + 参考光谱颜色校正 + λ鲁棒性验证（含指标导出）
%
% refImagePath      : 参考图路径（先从这张图估计参考 HERGB 光谱）
% inputDir          : 需要批量校正的图像文件夹（如含10张）
% outRoot           : 输出根目录（会在下面创建 correct/ 和 logs/）
% lambda_ortho_list : e.g. [1e-4 1e-3 1e-2]
% lambda_s_list     : e.g. [1e-5 1e-4 1e-3]
% varargin          : 名值对，可选参数：
%   'NumSample',300            随机采样像素数（非白非黑）
%   'ThrWhite',235             非白阈值
%   'LowCut',15                非黑阈值
%   'WantItersC',10            C步迭代轮数
%   'WantItersP',15            P步迭代轮数
%   'PatchVis',true            是否可视化采样点
%
% 输出：
%   - ./ref/ref_spectra.mat   （保存在当前路径/ref/）
%   - 每张图在 outRoot/correct/lmdO_xxx_lmdS_xxx/<imageName>/ 下保存：
%       RGB_orig / RGB_recon / Hematoxylin / Eosin / Diff_map
%       RGB_corrected (用参考光谱 + 本图的 cH/cE)
%   - outRoot/logs/metrics_<lambda>.csv 记录 PSNR/SSIM/NMI 等

%% ---------------- 参数与准备 ----------------
p = inputParser;
p.addParameter('NumSample', 2000);
p.addParameter('ThrWhite', 235);
p.addParameter('LowCut', 15);
p.addParameter('lambda_orthoC', 0);
p.addParameter('lambda_s', 0);
p.addParameter('PatchVis', true);
% p.addParameter('LambdaC', 1e-3);   % 浓度L1稀疏权重  λ_c
p.addParameter('Lambdax',0);   % 浓度互斥权重  λ_c
% p.addParameter('Rho',     6.0);    % ADMM 罚参数      ρ
p.addParameter('AdmmIters',50);   % 每次C步内 ADMM 迭代轮数


p.parse(varargin{:});
numSample  = p.Results.NumSample;
thrWhite   = p.Results.ThrWhite;
lowCut     = p.Results.LowCut;
lambda_ortho= p.Results.lambda_orthoC;
lambda_s= p.Results.lambda_s;
doPlot     = p.Results.PatchVis;
% lambda_c  = p.Results.LambdaC;
lambda_x  = p.Results.Lambdax;
% rho       = p.Results.Rho;
admmIters = p.Results.AdmmIters;


% if ~exist(outRoot,'dir'), mkdir(outRoot); end
% outCorrectRoot = fullfile(outRoot, 'correct'); if ~exist(outCorrectRoot,'dir'), mkdir(outCorrectRoot); end
% outLogRoot     = fullfile(outRoot, 'logs');    if ~exist(outLogRoot,'dir'), mkdir(outLogRoot); end

g5     = @(p) Gaussian_reconstruction5(p);
idxBlk = @(k) (15*(k-1)+1):(15*k);
Aidx   = [1 4 7 10 13];
muidx  = [2 5 8 11 14];
sidx   = [3 6 9 12 15];

% 优化选项
baseOpt = optimoptions('fmincon', ...
    'Algorithm','interior-point', ...
    'Display','off', ...
    'MaxIterations', 1e6, ...
    'MaxFunctionEvaluations', 1e8, ...
    'OptimalityTolerance', 1e-3, ...
    'StepTolerance', 1e-8);

% 像素子问题（lsqnonlin + 解析 Jacobian）
optsPixJac = optimoptions('lsqnonlin', ...
    'Display','off', ...
    'SpecifyObjectiveGradient', true, ...
    'MaxIterations', 50, ...
    'FunctionTolerance', 1e-8, ...
    'StepTolerance', 1e-8);
% =============估计参考光谱=================================================================
refDir = fullfile(pwd,'ref'); if ~exist(refDir,'dir'), mkdir(refDir); end
refMat = fullfile(refDir,'ref_spectra.mat');
refRangeMat = fullfile(refDir,'ref_concentration_range.mat');

needEstimateRefSpectra = true;
if exist(refMat, 'file')
    refData = load(refMat);
    needEstimateRefSpectra = ~all(isfield(refData, {'gH_ref','gE_ref','gR_ref','gG_ref','gB_ref','gMat_ref','den_ref'}));
    if ~needEstimateRefSpectra
        gH_ref = refData.gH_ref; gE_ref = refData.gE_ref;
        gR_ref = refData.gR_ref; gG_ref = refData.gG_ref; gB_ref = refData.gB_ref;
        gMat_ref = refData.gMat_ref; den_ref = refData.den_ref;
        fprintf('[REF] Loaded reference spectra from %s\n', refMat);
    end
end

if needEstimateRefSpectra
    fprintf('[REF] Estimating reference spectra from image: %s\n', refImagePath);
    lambda_temp_ref = 0;
    [gH_ref, gE_ref, gR_ref, gG_ref, gB_ref, gMat_ref, den_ref] = ...
        estimate_ref_spectra_from_image( ...
            refImagePath, numSample, thrWhite, lowCut, ...
            g5, idxBlk, Aidx, baseOpt, optsPixJac, ...
            lambda_ortho, lambda_s, lambda_temp_ref);
end

needEstimateRefRange = true;
if exist(refRangeMat, 'file') && ~needEstimateRefSpectra
    refRange = load(refRangeMat);
    needEstimateRefRange = ~all(isfield(refRange, {'refH_p95','refE_p95','refH_min','refH_max','refE_min','refE_max'}));
    if ~needEstimateRefRange
        refH_p95 = refRange.refH_p95; refE_p95 = refRange.refE_p95;
        refH_min = refRange.refH_min; refH_max = refRange.refH_max;
        refE_min = refRange.refE_min; refE_max = refRange.refE_max;
        fprintf('[REF] Loaded reference concentration range from %s\n', refRangeMat);
    end
end

if needEstimateRefRange
    fprintf('[REF] Estimating reference concentrations with fast GN, not MLP...\n');
    Iref = imread(refImagePath);
    Iref_od = rgb2od(double(Iref));
    optsRef = struct();
    optsRef.maxIter = 15;
    optsRef.lambda0 = 1e-2;
    optsRef.lambdaDecay = 0.5;
    optsRef.cMax = 4;
    optsRef.epsRGB = 1e-8;
    [cH_ref_full, cE_ref_full] = solve_concentration_fast_GN_patch(Iref_od, gH_ref, gE_ref, gMat_ref, den_ref, optsRef);
    cH_ref_full = cH_ref_full(:)';
    cE_ref_full = cE_ref_full(:)';
    refH_min = min(cH_ref_full); refH_max = max(cH_ref_full);
    refE_min = min(cE_ref_full); refE_max = max(cE_ref_full);
    refH_p95 = prctile(cH_ref_full, 95);
    refE_p95 = prctile(cE_ref_full, 95);
    save(refRangeMat, 'refH_p95', 'refE_p95', 'refH_min', 'refH_max', 'refE_min', 'refE_max');
    fprintf('[REF] Saved reference concentration range to %s\n', refRangeMat);
end

save(refMat, 'gH_ref', 'gE_ref', 'gR_ref', 'gG_ref', 'gB_ref', 'gMat_ref', 'den_ref', ...
    'refH_p95', 'refE_p95', 'refH_min', 'refH_max', 'refE_min', 'refE_max');
fprintf('[REF] Reference ready. P95 H=%.4f, P95 E=%.4f\n', refH_p95, refE_p95);

%% ---------------- 2) 遍历输入文件夹，做 λ 网格验证 + 输出 ----------------
imList = dir(fullfile(inputDir,'*.png'));

imList = [imList; dir(fullfile(inputDir,'*.jpg'))];
imList = [imList; dir(fullfile(inputDir,'*.tif'))];
imList = [imList; dir(fullfile(inputDir,'*.tiff'))];
if isempty(imList), error('输入文件夹没有找到图像。'); end
% Iref = imread(refImagePath);
% lambda_temp, alpha_hist, and alpha_update are runtime defaults.
% Their reported values should be determined by sweep experiments, not by the convergence proof.
lambda_temp = 1e-2;
alpha_update = 0.8;
alpha_hist = 0.001;

for io = 1:numel(C_list)
    for js = 1:numel(P_list)
        wantIters_c = C_list(io);
        wantIters_p    = P_list(js);
        % tag = sprintf('C_%g_P_%g',wantIters_c,wantIters_p);
        tag = sprintf('C_%g_P_%g', wantIters_c, wantIters_p);

        % metrics_csv = fullfile(outLogRoot, sprintf('metrics_%s.csv', strrep(tag,'.','p')));
        % fid = fopen(metrics_csv,'w');
        % fprintf(fid,'image,PSNR,SSIM,NMI,NMIv\n');

        for k =1:numel(imList)
            fname = imList(k).name;
            % % check
            % [~, fname_base, ~] = fileparts(fname);
            % [~, spectra_base, ~] = fileparts(spectra);
            % 
            % if ~strcmp(fname_base, spectra_base)
            %     error('File name mismatch: %s vs %s', fname, spectra);
            % end
            fprintf('\n[PROC] %s | %s\n', fname, tag);
            Ibig = imread(fullfile(inputDir,fname));
            % Ibig = load(fullfile(inputDir, fname));
            % I_od = Ibig.OD_rgb;          % m x n x 3
            % [m,n,~] = size(I_od);
            % P = reshape(I_od, [], 3)';   % 3 x (m*n)
            % N = m*n;
            I_rgb = double(Ibig);
            I_od  = rgb2od(I_rgb);
            [m,n,~] = size(Ibig);
            P = reshape(I_od, [],3)'; 
            N = m*n;

            % ===== 初值 =====
            % load p_fit5_result_H_global.mat p_fit; pH0 = p_fit(:);
            % load p_fit5_result_E_global.mat p_fit; pE0 = p_fit(:);
            % load p_fit5_result_R_global.mat p_fit; pR0 = p_fit(:);
            % load p_fit5_result_G_global.mat p_fit; pG0 = p_fit(:);
            % load p_fit5_result_B_global.mat p_fit; pB0 = p_fit(:);

            % ===== 定义加载路径 =====
            load_base_path = fileparts(mfilename('fullpath')); 
            
            % ===== 从路径中加载初值 =====
            % 加载 H
            data_H = load(fullfile(load_base_path, 'p_fit5_result_H_global.mat'), 'p_fit');
            pH0 = data_H.p_fit(:);
            
            % 加载 E
            data_E = load(fullfile(load_base_path, 'p_fit5_result_E_global.mat'), 'p_fit');
            pE0 = data_E.p_fit(:);
            
            % 加载 R
            data_R = load(fullfile(load_base_path, 'p_fit5_result_R_Singlegauss.mat'), 'p_fit');
            pR0 = data_R.p_fit(:);
            
            % 加载 G
            data_G = load(fullfile(load_base_path, 'p_fit5_result_G_Singlegauss.mat'), 'p_fit');
            pG0 = data_G.p_fit(:);
            
            % 加载 B
            data_B = load(fullfile(load_base_path, 'p_fit5_result_B_Singlegauss.mat'), 'p_fit');
            pB0 = data_B.p_fit(:);


            bH = idxBlk(1); bE = idxBlk(2);
            bR = idxBlk(3); bG = idxBlk(4); bB = idxBlk(5);

            p0 = [pH0; pE0; pR0; pG0; pB0];
            
            % ===== 6. 设置主循环的约束 lb / ub =====
            lb = -inf(size(p0));
            ub =  inf(size(p0));
            
            % 通用约束 A, sigma
            for kk = 1:5
                bi = idxBlk(kk);
                lb(bi(Aidx)) = 0;    ub(bi(Aidx)) = 1;
                lb(bi(sidx)) = 1e-3;
            end
            
            % --- 约束 H / E ---
            % 策略：如果你的 H/E 库非常准，可以锁死；如果想微调，给一点空间
            % 这里沿用你的逻辑：固定 H/E (Warm-up 阶段建议固定，后续可放开)
            % lb(bH) = pH0; ub(bH) = pH0;
            % lb(bE) = pE0; ub(bE) = pE0;
            
            % --- 约束 R / G / B ---
            % % 使用刚才校准用的物理范围
            % lb(bR(muidx)) = 149;    ub(bR(muidx)) = 301;
            % lb(bG(muidx)) = 100;    ub(bG(muidx)) = 200;
            % lb(bB(muidx)) = 1;      
            % ub(bB(muidx)) = 151;

            % 添加的RGB先验
            % --- RGB sigma 物理范围 ---
            lb(bR(sidx)) = 10;    ub(bR(sidx)) = 80;
            lb(bG(sidx)) = 10;    ub(bG(sidx)) = 80;
            lb(bB(sidx)) = 10;    ub(bB(sidx)) = 80;

            lb(bB(muidx)) = 20;     ub(bB(muidx)) = 130;   % 420-530 nm
            lb(bG(muidx)) = 90;     ub(bG(muidx)) = 210;   % 490-610 nm
            lb(bR(muidx)) = 150;    ub(bR(muidx)) = 290;   % 550-690 nm
            
            % 确保初值在边界内
            p0 = project_to_open_box(p0, lb, ub);

            % ===== 随机采样像素用于外层交替优化（加速）=====
            [P_s, ~, Ns, ~, ~, ~] = samplePixelsNonwhite(Ibig, Ibig, numSample, thrWhite, 42, false, lowCut);


            p_cur = p0;

            % 初始 C步
            pH=p_cur(1:15); pE=p_cur(16:30);
            pR=p_cur(31:45); pG=p_cur(46:60); pB=p_cur(61:75);
            gH=g5(pH); gE=g5(pE); gR=g5(pR); gG=g5(pG); gB=g5(pB);
            gH_ini = gH; gE_ini = gE;
            gMat=[gR(:),gG(:),gB(:)]; den=[sum(gR);sum(gG);sum(gB)];
            cH=zeros(1,Ns); cE=zeros(1,Ns);

            % % ---- Warm start for ADMM (after initial cH,cE are computed) ----
            % zH = cH; 
            % zE = cE;
            % uH = zeros(1, Ns); 
            % uE = zeros(1, Ns);

                        % ---- 外层交替：C-step -> P-step ----
            maxOuter = 60;                % 最大外层轮数
            
            % --- 收敛监控参数 ---
            minOuter = 5;               % 最少外层轮数 (防止早停)
            tol_relP = 1e-5;              % P 的相对变化容忍
            tol_relC = 1e-4;              % C 的相对变化容忍

             % ===== C-step: L2 正则权重 =====
            lambda2 = 1e-3;   % 建议先试 1e-3 或 3e-3，可调

            % --- 历史记录容器 ---
            J_total_hist   = zeros(maxOuter, 1);
            delta_P_hist   = zeros(maxOuter, 1);
            delta_C_hist   = zeros(maxOuter, 1);
            
            % --- P-step (fmincon) 的超参数 ---
            wantIters_p = 2; % 关键：P-step 是非精确的
            L_P_current = 1e-3;   % 初始 γ_p / Lipschitz 近似，可视作经验 L_P

            optIterP = optimoptions(baseOpt,'OutputFcn',stopAtIterFactory(wantIters_p));
            
            % --- C-step (ADMM) 的超参数 ---
            % tau0 = 1e-1;     % MM 稳定化强度
            % alpha_hist   = 1e-1;   % 强度（建议：1e-2 ~ 1e-1）
            % alpha_update = 0.5;     % 最佳：0.2 ~ 0.5

            
            % BCD 循环开始
            fprintf('[BCD START] MaxIters=%d, P-Iters=%d, C-ADMM-Iters=%d\n', maxOuter, wantIters_p, admmIters);
            
            % 初始化 "k-1" (上一轮) 的解
            p_k_minus_1 = p_cur;
            cH_k_minus_1 = cH;
            cE_k_minus_1 = cE;
            % 初始化历史浓度（非常重要）
            cH_hist = zeros(size(cH));
            cE_hist = zeros(size(cE));

            % zH_k_minus_1 = zH;
            % zE_k_minus_1 = zE;
            % uH_k_minus_1 = uH;
            % uE_k_minus_1 = uE;

            % lambda_temp = 5e-2;

           % ===== Best-so-far tracking =====
            bestJ   = inf;
            bestIt  = 0;
            bestP   = p_cur;     % 或者 p_k_minus_1 / p_init
            bestcH  = cH;
            bestcE  = cE;
            % 
            % tol_relP = 1e-5;
            % tol_relJ = 1e-4;
            patience = 5;
            % noImproveCnt = 0;
            usePatience = true;

            for it = 1:maxOuter
                % if it <= 10
                %     wantIters_p = 10;
                %     L_P_current = 1e-6;
                %     % lambda_temp_cur = 0;
                % else
                %     wantIters_p = 3;
                %     L_P_current = 1e-3;
                %     % lambda_temp_cur = lambda_temp;
                % end
                % optIterP = optimoptions(baseOpt,'OutputFcn',stopAtIterFactory(wantIters_p));
                
                % ---------- (A) C-step（求解 C_k） ----------
                % 使用 (k-1) 轮的 P 和 Z/U 计算
                
                [gH,gE,gR,gG,gB,gMat,den] = get_gs_from_p(p_k_minus_1,g5);


                [cH, cE, cH_hist, cE_hist] = solve_C_smooth_history( ...
                        P_s, gH, gE, gMat, den, ...
                        cH, cE, ...
                        cH_hist, cE_hist, ...
                        optsPixJac, ...
                        alpha_hist, alpha_update );

                % [cH, cE, cH_hist, cE_hist, scaleInfo] = normalize_C_pixelwise_p95( ...
                %     cH, cE, cH_hist, cE_hist);
                % 
                % if scaleInfo.used
                %     fprintf('C pixelwise norm: p95H=%.3e, p95E=%.3e, valid=%d\n', ...
                %         scaleInfo.p95H, scaleInfo.p95E, scaleInfo.numValid);
                % end


                
                % ---------- (B) P-step（求解 P_k） ----------
                % 定义 P-step 的目标函数
                % ★ 关键：costP 使用 C-step *刚刚计算出* 的 zH, zE
                
                %   P-step: Lipschitz Backtracking Proximal MM（推荐论文写法）
                % ============================================================
                % 
                % ---------- (B) P-step（求解 P_k） ----------
                % 使用 backtracking 选择 γ_p = L，使得满足 MM 上界条件
                % F_core(p) = P-block 的真实目标（不含 proximal）
                % F_core = @(p_all) local_cost_with_fixed_c_reg( ...
                %                     p_all, P_s, zH, zE, g5, ...
                %                     lambda_ortho, lambda_s, lambda_temp, ...
                %                     gH_ref, gE_ref);

                
                % 
                % optsRefit = struct();
                % optsRefit.maxIter = 2;          % P-step 内部只做 1~2 次
                % optsRefit.lambda0 = 1e-2;
                % optsRefit.lambdaDecay = 0.5;
                % optsRefit.lambdaMin = 1e-8;
                % optsRefit.cMax = 10;
                % optsRefit.epsRGB = 1e-8;
                % optsRefit.chunkSize = 500000;
                % 
                % % 用当前 C 作为 refit 初值
                % optsRefit.cH_init = cH(:);
                % optsRefit.cE_init = cE(:);
                % 
                % % 如果要让光程先验影响 P-step，就在 refit 里启用
                % optsRefit.sigma_t = 0.3;
                % optsRefit.pathTarget = 1.0;
                % optsRefit.p95Floor = 1e-2;
                % 
                % F_core = @(p_all) local_cost_with_fast_C_refit( ...
                %                     p_all, P_s, cH, cE, g5, ...
                %                     optsRefit, ...
                %                     lambda_ortho, lambda_s, lambda_temp, ...
                %     gH_ini, gE_ini);

                 F_core = @(p_all) local_cost_with_fixed_c_reg( ...
                                    p_all, P_s, cH, cE, g5, ...
                                    lambda_ortho, lambda_s, lambda_temp, ...
                                    gH_ini, gE_ini);
                 % 
                 % F_core = @(p_all) local_cost_with_fixed_c_reg( ...
                 %                    p_all, P_s, cH, cE, g5, ...
                 %                    lambda_ortho, lambda_s, lambda_temp, ...
                 %                    gH_ini, gE_ini);
                
                L_try = L_P_current;       % 从上一轮的 L 开始试
                maxBT = 5;                 % backtracking 最多尝试次数，防止极端情况死循环
                bt_cnt = 0;
                
                F_old = F_core(p_k_minus_1);   % F(P^{k-1})，用于 MM 条件比较
                
                while true
                    gamma_p = L_try;   % 本轮 proximal 强度 = 当前试探的 L
                
                    % 定义带 proximal 的 P-step 目标：F_core + (gamma_p/2)||p - p_prev||^2
                    costP = @(p_all) F_core(p_all) ...
                              + 0.5 * gamma_p * sum((p_all - p_k_minus_1).^2);
                
                    % 用 fmincon 做“非精确” P-step，只跑 wantIters_p 次
                    % [p_trial, Jp] = fmincon(costP, p_k_minus_1, [], [], [], [], ...
                    %                         lb, ub, @nonlcon_physical_HE_RGB_relation, optIterP);

                    [p_trial, Jp] = fmincon(costP, p_k_minus_1, [], [], [], [], ...
                        lb, ub, @nonlcon_physical_HE_RGB_relation, optIterP);
                
                    % 计算真实 P-block 目标（不含 proximal 项），用于 MM 条件比较
                    F_new = F_core(p_trial);
                    dP    = p_trial - p_k_minus_1;
                    rhs   = F_old + 0.5 * L_try * sum(dP.^2);
                
                    % ---- MM 上界条件检查：F_new <= F_old + (L/2)||p_new - p_prev||^2 ----
                    if F_new <= rhs
                        % 条件满足：说明 L_try 是一个合法的 Lipschitz 上界
                        p_cur       = p_trial;
                        L_P_current = L_try;   % 记录下来，下次从这个 L 开始尝试
                        break;
                    else
                        % 条件不满足：L_try 太小了，放大再试
                        L_try = L_try * 5;
                        bt_cnt = bt_cnt + 1;
                        if bt_cnt >= maxBT
                            % 保险起见：超过最大 backtracking 次数，就接受当前结果
                            % （通常不会触发）
                            warning('P-step backtracking reached maxBT; accepting current p_trial.');
                            p_cur       = p_trial;
                            L_P_current = L_try;
                            break;
                        end
                    end
                end
                % costP = @(p_all) local_cost_with_fixed_c_reg(p_all, P_s, zH, zE, g5, lambda_ortho, lambda_s,lambda_temp, gH_ref, gE_ref) ...
                %                  + 0.5 * gamma_p * sum((p_all - p_k_minus_1).^2);
                % 
                % % P-step 求解 (只迭代 wantIters_p 次)
                % % 热启动点是 p_k_minus_1
                % [p_cur, Jp] = fmincon(costP, p_k_minus_1, [], [], [], [], lb, ub, ...
                %                       @nonlcon_physical_HE_RGB_relation, optIterP);


                % ====== (B') 每一轮 P-step 后进行光谱归一化与浓度重标定 ======
                % 
                % 先根据 p_cur 得到当前光谱
                [gH_tmp, gE_tmp, gR_tmp, gG_tmp, gB_tmp, ~, ~] = get_gs_from_p(p_cur, g5);

                % ------------------------------------------------------
                % 1. H 通道最大值归一化
                %    目标：max(gH) = 1
                %    补偿：cH = cH * sH
                % ------------------------------------------------------
                sH = max(gH_tmp);
                
                if sH > 1e-9
                    % bH = idxBlk(1);
                
                    % 归一化 H 光谱参数
                    % 注意：这里假设 Aidx 对应的是 H 光谱的幅值参数
                    p_cur(bH(Aidx)) = p_cur(bH(Aidx)) / sH;
                
                    % 反向补偿 H 浓度
                    % 因为 gH -> gH/sH，所以 cH -> cH*sH，保持 gH*cH 不变
                    cH = cH * sH;
                
                    % 历史变量也必须同步缩放
                    if exist('cH_hist', 'var')
                        cH_hist = cH_hist * sH;
                    end
                
                    % 如果你的代码里还在使用 zH/uH，也要同步缩放
                    if exist('zH', 'var')
                        zH = zH * sH;
                    end
                    if exist('uH', 'var')
                        uH = uH * sH;
                    end
                end
                
                % ------------------------------------------------------
                % 2. E 通道最大值归一化
                %    目标：max(gE) = 1
                %    补偿：cE = cE * sE
                % ------------------------------------------------------
                sE = max(gE_tmp);
                
                if sE > 1e-9
                    % bE = idxBlk(2);
                
                    % 归一化 E 光谱参数
                    p_cur(bE(Aidx)) = p_cur(bE(Aidx)) / sE;
                
                    % 反向补偿 E 浓度
                    cE = cE * sE;
                
                    % 历史变量也必须同步缩放
                    if exist('cE_hist', 'var')
                        cE_hist = cE_hist * sE;
                    end
                
                    % 如果你的代码里还在使用 zE/uE，也要同步缩放
                    if exist('zE', 'var')
                        zE = zE * sE;
                    end
                    if exist('uE', 'var')
                        uE = uE * sE;
                    end
                end                

                % ------------------------------------------------------
                % 3. Independent RGB channel normalization
                %    max(gR) = max(gG) = max(gB) = 1
                % ------------------------------------------------------
                rgb_profiles = {gR_tmp(:), gG_tmp(:), gB_tmp(:)};

                for cc = 1:3
                    gC = rgb_profiles{cc};
                    kappaC = max(gC);

                    if kappaC > 1e-9
                        bC = idxBlk(cc + 2);   % blocks 3,4,5 -> R,G,B
                        p_cur(bC(Aidx)) = p_cur(bC(Aidx)) / kappaC;
                    end
                end

             % ---------- (C) 监控与记录 ----------

                % (1) 计算 J_total (用于监控 + 选择 best)
                [gH, gE, gR, gG, gB, gMat, den] = get_gs_from_p(p_cur, g5);
                
                [cos2_val, sH, sE] = reg_terms(gH, gE);
                [Jfid, ~] = datafid_and_l1(P_s, gH, gE, gMat, den, cH, cE, 0);
                
                % 这里用你真正想比较的目标（你现在用的是 Jfid）
                J_total = F_core(p_cur);
                
                J_total_hist(it) = J_total;
                
                % ===== (1.5) 更新 best-so-far =====
                if J_total < bestJ
                    bestJ  = J_total;
                    bestIt = it;
                    bestP  = p_cur;
                    bestcH = cH;
                    bestcE = cE;
                    noImproveCnt = 0;   % patience 计数清零
                else
                    noImproveCnt = noImproveCnt + 1;
                end
                
                % (2) 计算解的相对变化量 (Delta)
                delta_P = norm(p_cur - p_k_minus_1) / (norm(p_k_minus_1) + 1e-8);
                delta_C = norm([cH - cH_k_minus_1, cE - cE_k_minus_1], 'fro') / ...
                          (norm([cH_k_minus_1, cE_k_minus_1], 'fro') + 1e-8);
                
                delta_P_hist(it) = delta_P;
                delta_C_hist(it) = delta_C;
                
                fprintf('[BCD Iter %02d/%02d] J=%.4e (best=%.4e@%d), dP=%.3e, dC=%.3e\n', ...
                        it, maxOuter, J_total, bestJ, bestIt, delta_P, delta_C);
                
                % (3) 检查收敛（早停）
                stopOK = false;
                
                if it >= minOuter
                    % 方案A：保持你原来的“解变化小”早停
                    if delta_P < tol_relP && delta_C < tol_relC
                        if usePatience
                            % 方案B：再要求 loss 最近 patience 轮几乎不改善（更稳）
                            if noImproveCnt >= patience
                                stopOK = true;
                            end
                        else
                            stopOK = true;
                        end
                    end
                end
                
                if stopOK
                    disp('Early stop triggered. Will return best-so-far solution.');
                    break;
                end

                
                % (4) 准备下一轮 k-1 变量
                p_k_minus_1 = p_cur;
                cH_k_minus_1 = cH;
                cE_k_minus_1 = cE;
                % zH_k_minus_1 = zH;
                % zE_k_minus_1 = zE;
                % uH_k_minus_1 = uH;
                % uE_k_minus_1 = uE;

            end % 结束 BCD (Outer) 循环
            p_opt = bestP;
            cH = bestcH;
            cE = bestcE;
  % --- 5. 截断历史记录 (Cleanup history arrays) ---
            % (it 是 BCD 循环停止时的迭代次数)
            J_total_hist = J_total_hist(1:it);
            delta_P_hist = delta_P_hist(1:it);
            delta_C_hist = delta_C_hist(1:it);

            % (A) 创建子目录
            subDir = fullfile(outRoot, 'path0', erase(fname, '.png'));
            if ~exist(subDir,'dir'), mkdir(subDir); end

            % --- 6. 应用：全图浓度计算
            % --- 6. 应用：全图浓度计算 (使用神经网络加速) ---
            fprintf('BCD 结束。正在使用最终光谱训练网络并预测全图浓度...\n');
            [gH_opt, gE_opt, gR_opt, gG_opt, gB_opt, gMat_opt, den_opt] = get_gs_from_p(p_opt, g5);
            
            % 归一化 (保持与原逻辑一致)
            sH = max(gH_opt); gH_opt = gH_opt./sH;
            sE = max(gE_opt); gE_opt = gE_opt./sE;
            
            %% 调用神经网络函数直接预测全图
            % 输入: I_od (从前面代码看是 rgb2od 算出来的，或者 reshape 之前的 P)
            % 注意：您的 P 是 3xN，我的函数需要 N x 3 或 m x n x 3
            % 
            % % 这里我们直接传 P' (N x 3) 进去
            % [cH_full, cE_full] = train_and_predict_concentration(P', gH_opt, gE_opt, gMat_opt, den_opt,subDir,fname);
            

            %% FAST GN 预测全图

            opts = struct();
            opts.maxIter = 15;       % 8~15 一般够
            opts.lambda0 = 1e-2;     % 保守一点更稳
            opts.lambdaDecay = 0.5;  % 阻尼衰减
            opts.cMax = 4;         % 如果你担心爆点，可以设 5 或 8
            opts.epsRGB = 1e-8;
            [cH_full, cE_full] = solve_concentration_fast_GN_patch(P', gH_opt, gE_opt, gMat_opt, den_opt, opts);

            %% 稳定版本FAST GN 预测全图
            % % ---------- Pass 1: loose estimation ----------
            % opts1 = struct();
            % opts1.maxIter = 15;
            % opts1.lambda0 = 1e-2;
            % opts1.lambdaDecay = 0.5;
            % opts1.lambdaGrow = 5;
            % opts1.lambdaMin = 1e-8;
            % opts1.lambdaMax = 1e6;
            % 
            % opts1.cMax = 4;
            % opts1.maxStep = 0.3;
            % 
            % opts1.epsRGB = 1e-8;
            % opts1.tinyToZero = 1e-6;
            % opts1.chunkSize = 300000;
            % opts1.useMultiInit = true;
            % opts1.postPctlClip = false;
            % opts1.verbose = false;
            % 
            % [cH0, cE0] = solve_concentration_fast_GN_patch_stable0610(P', gH, gE, gMat, den, opts1);
            % 
            % % ---------- Estimate adaptive upper bound ----------
            % vH = cH0(isfinite(cH0) & cH0 > 0);
            % vE = cE0(isfinite(cE0) & cE0 > 0);
            % 
            % pH = prctile(vH, 99.5);
            % pE = prctile(vE, 99.5);
            % 
            % cMax_auto = max(pH, pE);
            % 
            % % 防止过小或过大
            % cMax_auto = max(cMax_auto, 1);
            % cMax_auto = min(cMax_auto, 10);
            % 
            % fprintf('Auto cMax = %.4f\n', cMax_auto);
            % 
            % % ---------- Pass 2: constrained refinement ----------
            % opts2 = opts1;
            % opts2.maxIter = 20;
            % opts2.cMax = cMax_auto;
            % opts2.maxStep = 0.1 * cMax_auto;
            % opts2.postPctlClip = false;
            % 
            % [cH_full, cE_full] = solve_concentration_fast_GN_patch_stable0610(P', gH, gE, gMat, den, opts2);

            %% 使用逐点估计c的方式估计全图c
           % optsPixJac = optimoptions('lsqnonlin', ...
           %      'Display', 'off', ...
           %      'SpecifyObjectiveGradient', true, ...
           %      'MaxIterations', 100, ...
           %      'FunctionTolerance', 1e-12, ...
           %      'StepTolerance', 1e-12, ...
           %      'OptimalityTolerance', 1e-12);
           % 
           %  useParallel = true;
           % 
           %  cH = zeros(1,m*n);cE = zeros(1,m*n);
           %  [cH_full, cE_full, ~] = solve_C_nonlinear_lsq_no_cmax( ...
           %      P, gH, gE, gMat, den, ...
           %      cH, cE, ...
           %      optsPixJac, useParallel);

                        
            % 转置回您代码需要的 1 x N 格式
            cH_full = cH_full';
            cE_full = cE_full';
            
            fprintf('全图计算完成 (Neural Network Accelerated)。\n');

            % --- 7. 浓度归一化 (来自 BKSVD 流程) ---
            cH_ref = refH_p95; % 示例目标 P95 值
            cE_ref = refE_p95; 
            
            alphaH = cH_ref / (prctile(cH_full, 95) + 1e-8);
            alphaE = cE_ref / (prctile(cE_full, 95) + 1e-8);
            
            cH_full_norm = cH_full * alphaH;
            cE_full_norm = cE_full * alphaE;

            % --- 8. 生成重建图和差异图 ---
            
            % (A) 重建OD图 (使用本图优化后的光谱)
            % (假设 forward_od 返回 [m, n, 3])
            OD_recon= -log10( (gMat_opt.' * 10.^(-(gH_opt*cH_full + gE_opt*cE_full))) ./ den_opt );
            OD_recon = reshape(OD_recon.', m,n, 3);
            % OD_recon = forward_od(gH_opt, gE_opt, gMat_opt, den_opt, cH_full, cE_full, m, n);
            RGB_recon = uint8(od2rgb(OD_recon)); % 重建的RGB
            
            % (B) OD 差异图
            % ★ 关键修复 A: OD_orig 应该是 [m, n, 3] 的 I_od
            % (I_od 是在 BCD 循环 *之前* 定义的)
            OD_orig = I_od; 
            OD_diffmap = OD_orig - OD_recon; % 两个 [m, n, 3] 矩阵相减
            OD_diffmap =  uint8(od2rgb(OD_diffmap)); 
            
            % % (C) 颜色校正图 (使用参考光谱)
            % % (确保 refMat 包含这些变量)
            % load(refMat, 'gH_ref', 'gE_ref', 'gR_ref', 'gG_ref', 'gB_ref', 'den_ref');
            % gMat_ref = [gR_ref(:), gG_ref(:), gB_ref(:)];
            % 
            % % 使用归一化的浓度 + 参考光谱
            % OD_corr_norm = forward_od(gH_ref, gE_ref, gMat_ref, den_ref, cH_full_norm, cE_full_norm, m, n);
            % RGB_corr_norm = uint8(od2rgb(OD_corr_norm)); % 最终校正结果
            % (C) 颜色校正图 (使用参考光谱)
            % 这里使用前面加载/估计的 gH_ref, gE_ref, gR_ref, gG_ref, gB_ref, den_ref
            gMat_ref = [gR_ref(:), gG_ref(:), gB_ref(:)];
            
            % 使用“归一化后的浓度”做颜色校正
            OD_corr_norm = forward_od(gH_ref, gE_ref, gMat_ref, den_ref, ...
                                      cH_full_norm, cE_full_norm, m, n);
            RGB_corr_norm = uint8(od2rgb(OD_corr_norm+OD_orig - OD_recon)); % 最终校正结果

            
            % --- 9. 输出保存 (重构版本) ---
            
            
            
            % (B) 保存主要图像结果
            imwrite(uint8(od2rgb(I_od)), fullfile(subDir, '01_RGB_orig.png'));
            imwrite(RGB_recon, fullfile(subDir, '02_RGB_recon_self.png'));
            imwrite(OD_diffmap, fullfile(subDir, '03_RGB_diff.png'));
            imwrite(RGB_corr_norm,      fullfile(subDir, '04_RGB_corrected_ref.png'));  % ★ 新增：参考图颜色校正结果
            f_err = figure('Visible','off', 'Position', [100, 100, 1000, 450]);
            errOD_img = OD_diffmap./uint8(od2rgb(I_od));
            imagesc(errOD_img); axis image off; colorbar;
            meanerror = mean(errOD_img);
            title(sprintf('OD error Mean=%.3f', meanerror));
            saveas(f_err, fullfile(subDir, 'OD_Error_Heatmaps.png'));
            close(f_err);
            
            % (C) 保存 .mat 数据 (用于定量比较)
            save(fullfile(subDir, 'results_data.mat'), 'p_opt', 'cH_full', 'cE_full', 'cH_full_norm', 'cE_full_norm');
            
            % (D) 保存监控曲线 (J_total, delta_P, delta_C)
            f_conv = figure('Visible','off', 'Position', [100, 100, 600, 900]);
            
            subplot(3,1,1);
            plot(1:it, J_total_hist(1:it), 'b-o', 'LineWidth', 1.5); grid on;
            title(sprintf('J_total (Final: %.4e)', J_total_hist(it)));
            ylabel('J total');
            
            subplot(3,1,2);
            semilogy(1:it, delta_P_hist(1:it), 'r-o', 'LineWidth', 1.5); grid on;
            title(sprintf('Delta_P (Final: %.3e)', delta_P_hist(it)));
            ylabel('log(Delta P)');
            
            subplot(3,1,3);
            semilogy(1:it, delta_C_hist(1:it), 'g-o', 'LineWidth', 1.5); grid on;
            title(sprintf('Delta_C (Final: %.3e)', delta_C_hist(it)));
            ylabel('log(Delta C)');
            xlabel('Outer Iteration (k)');
            
            saveas(f_conv, fullfile(subDir, 'A_Convergence_Curves.png'));
            close(f_conv);
            
           % (E) 保存浓度图 
                       % ===== H/E 可视化 =====
            OD_H = forward_od(gH_opt,gE_opt,gMat_opt,den_opt,cH_full,zeros(1,N),m,n);
            OD_E = forward_od(gH_opt,gE_opt,gMat_opt,den_opt,zeros(1,N),cE_full,m,n);
            imwrite(uint8(od2rgb(OD_H)), fullfile(subDir,'Hematoxylin.png'));
            imwrite(uint8(od2rgb(OD_E)), fullfile(subDir,'Eosin.png'));

            % % --- (F) 浓度 vs Ground Truth 对比 ---------------------------------
            % % 假设 GT_concentrations.mat 中包含 cH_gt, cE_gt
            % gt_conc = load('GT_concentrations.mat');
            % cH_gt = gt_conc.GT_C_H;
            % cE_gt = gt_conc.GT_C_E;
            % 
            % % 统一成 1 x N 形式
            % if numel(cH_gt) ~= N
            %     error('GT concentration size mismatch: expected %d elements, got %d.', N, numel(cH_gt));
            % end
            % cH_gt_vec = reshape(cH_gt, 1, []);
            % cE_gt_vec = reshape(cE_gt, 1, []);
            % 
            % % === 额外：scale-invariant RMSE（允许整体乘一个alpha） ===
            % numH = sum(cH_full .* cH_gt_vec);
            % denH = sum(cH_full.^2) + 1e-8;
            % alphaH_opt = numH / denH;                % 最优缩放因子
            % 
            % 
            % errH   = cH_full - cH_gt_vec;
            % 
            % RMSE_H = sqrt(mean(errH.^2));
            % MAE_H  = mean(abs(errH));
            % 
            % metrics_conc.RMSE_H = RMSE_H;
            % metrics_conc.MAE_H  = MAE_H;
            % metrics_conc.Alpha_H   = alphaH_opt;
            % 
            % numE = sum(cE_full .* cE_gt_vec);
            % denE = sum(cE_full.^2) + 1e-8;
            % alphaE_opt = numE / denE;                % 最优缩放因子
            % 
            % 
            % errE   = cE_full - cE_gt_vec;
            % 
            % RMSE_E = sqrt(mean(errE.^2));
            % MAE_E  = mean(abs(errE));
            % 
            % metrics_conc.RMSE_E_SI = RMSE_E;
            % metrics_conc.MAE_E_SI  = MAE_E;
            % metrics_conc.Alpha_E   = alphaE_opt;
            % % 误差热图（H/E 各一张）：用归一化浓度的误差
            % errH_img = reshape(errH, m, n);
            % errE_img = reshape(errE, m, n);
            % 
            % f_err = figure('Visible','off', 'Position', [100, 100, 1000, 450]);
            % subplot(1,2,1);
            % imagesc(errH_img); axis image off; colorbar;
            % title(sprintf('H error RMSE=%.3f', RMSE_H));
            % 
            % subplot(1,2,2);
            % imagesc(errE_img); axis image off; colorbar;
            % title(sprintf('E error RMSE=%.3f', RMSE_E));
            % 
            % saveas(f_err, fullfile(subDir, 'D_Concentration_Error_Heatmaps.png'));
            % close(f_err);

            % 
            % (G) 保存 BCD 历史 (用于调试)
            histos = struct();
            histos.J_total_hist = J_total_hist;
            histos.delta_P_hist = delta_P_hist;
            histos.delta_C_hist = delta_C_hist;
            save(fullfile(subDir,'converge_hist.mat'),'-struct','histos');
            
            % (到此结束，接 "end % 结束 k (图像) 循环")


            % % ===== 指标（和你之前一致思路，示例：与原图/或交叉）=====
            % % 这里用 recon vs orig 的 SSIM/PSNR，NMI 按你之前“灰度均值比例”口径
            % PSNRv = psnr(RGB_corr, RGB_orig);
            % SSIMv = ssim(rgb2gray(RGB_corr), rgb2gray(RGB_orig));
            % Imean_recon = mean(RGB_corr,3); Imean_ref = mean(Iref,3);
            % nmi1 = median(Imean_recon(Imean_recon>0))/prctile(Imean_recon(Imean_recon>0),95);
            % nmi2 = median(Imean_ref (Imean_ref >0))/prctile(Imean_ref (Imean_ref >0),95);
            % NMIv = nmi1/nmi2 - 1;
            % fprintf(fid,'%s,%.6f,%.6f,%.6f,%.6f\n', fname, PSNRv, SSIMv, nmi1, NMIv);
        end
        % fclose(fid);
        % fprintf('[LOG] 指标已写入：%s\n', metrics_csv);
    end
end

fprintf('\n✅ ALL DONE. \n');
end

% end

%% ===================== 工具/子函数区 =====================

function p0 = project_to_open_box(p0, lb, ub)
    tol = 1e-8;
    p0 = max(p0, lb); p0 = min(p0, ub);
    finiteL = isfinite(lb); finiteU = isfinite(ub);
    p0(finiteL) = max(p0(finiteL), lb(finiteL)+tol);
    p0(finiteU) = min(p0(finiteU), ub(finiteU)-tol);
end

function OD_img = forward_od(gH, gE, gMat, den, cH, cE, m, n)
    % gH,gE : 300x1
    % gMat  : 300x3
    % den   : 3x1
    % cH,cE : 1xN  (N = m*n)
    % 返回  : m x n x 3 的 OD 图像

    den  = max(den, 1e-12);

    expo = -(gH*cH + gE*cE);     % 300 x N
    Y    = 10.^(expo);            % 300 x N
    T    = gMat.' * Y;           % 3   x N
    T    = max(T, 1e-12);

    OD = -log10(bsxfun(@rdivide, T, den));   % 3 x N
    OD_img = reshape(OD.', m, n, 3);         % N→(m,n), 通道最后
end


function [r, J] = pix_residual_with_jac(x, Pj, gH, gE, Gmat, den)
    den  = max(den, 1e-12);
    S  = gH*x(1) + gE*x(2);      % 300×1
    Y  = 10.^(-S);                % 300×1
    T  = Gmat.' * Y;             % 3×1
    T  = max(T, 1e-12);
    F  = -log10( T ./ den );     % 3×1
    r  = F - Pj;
    SH = Gmat.' * (gH .* Y);     % 3×1
    SE = Gmat.' * (gE .* Y);     % 3×1
    JH = SH ./ T;
    JE = SE ./ T;
    J  = [JH, JE];               % 3×2
end
function [r, J] = pix_residual_with_jac_quantify(x, Pj, gH, gE, Gmat, den)
    % 输入:
    % x: [cH; cE] 当前浓度
    % Pj: 目标像素的 OD 值 (3x1)
    % gH, gE: 吸收光谱 (300x1)
    % Gmat: 相机灵敏度矩阵 (300x3)
    % den: 归一化分母 (3x1)
    
    den  = max(den, 1e-12); % 防止分母为0

    % --- A. 正向物理模型 (连续域) ---
    S  = gH*x(1) + gE*x(2);      % 300x1, 总吸收
    Y  = 10.^(-S);               % 300x1, 连续透射光谱
    T_cont = Gmat.' * Y;         % 3x1,   连续相机响应 (未归一化)
    T_cont = max(T_cont, 1e-12); % 物理约束

    % --- B. 模拟 WSI 扫描仪的量化过程 (核心修改) ---
    % 1. 归一化到 0-1
    Signal_norm = T_cont ./ den;
    
    % 2. 量化到 0-255 整数 (模拟相机/扫描仪输出)
    RGB_int = round(255 * Signal_norm); 
    
    % 3. 转回连续域用于计算 OD (Dequantization)
    % 注意：防止 RGB_int 为 0 导致 log(0) 无穷大
    % 实际扫描仪中最黑通常也有 1/255 或电子噪声，这里设置下限
    RGB_quantized = max(RGB_int, 1) / 255; 
    
    % 4. 计算量化后的 OD 值
    OD_quantized = -log10(RGB_quantized);

    % --- C. 计算残差 (Residual) ---
    % 优化目标是让量化后的 OD 接近真实测量的 OD
    r = OD_quantized - Pj;

    % --- D. 计算 Jacobian (使用连续模型近似) ---
    % 注意：由于 round() 不可导，我们必须返回“连续模型”的 Jacobian
    % 来为优化器提供正确的下降方向。
    
    % 辅助变量
    SH = Gmat.' * (gH .* Y);     % 3x1
    SE = Gmat.' * (gE .* Y);     % 3x1
    
    % Jacobian of the broadband OD model. The ln(10) factors cancel,
    % yielding dF/dx_H = SH./T_cont and dF/dx_E = SE./T_cont.
    
    JH = SH ./ T_cont;
    JE = SE ./ T_cont;
    
    J  = [JH, JE];               % 3x2
end
function S = local_cost_with_fixed_c_reg(p_all, P, cH, cE, g5, ...
                                         lambda_ortho, lambda_s, ...
                                         lambda_temp, gH_ref, gE_ref)
    % ---------------------------------------------------------------
    % p_all       : 5 blocks of Gaussian params (H,E,R,G,B)
    % P           : 3×Ns sampled OD
    % cH,cE       : 1×Ns fixed concentrations (here usually zH,zE)
    % g5          : handle to Gaussian_reconstruction5
    % lambda_ortho, lambda_s : 原有正交/平滑权重（可为 0）
    % lambda_temp : 模板正则权重 (e.g. 1e-3 ~ 1e-1)
    % gH_ref,gE_ref : reference spectra (same length as gH,gE)
    % S           : scalar cost
    % ---------------------------------------------------------------

    idx = @(k) (15*(k-1)+1):(15*k);

    % ---- 拆出各个染料/相机的参数块 ----
    pH = p_all(idx(1));  
    pE = p_all(idx(2));
    pR = p_all(idx(3));  
    pG = p_all(idx(4));  
    pB = p_all(idx(5));

    % ---- 由参数生成光谱 ----
    gH = g5(pH); 
    gE = g5(pE);
    gR = g5(pR); 
    gG = g5(pG); 
    gB = g5(pB);

    % ---- 相机矩阵与归一化因子 ----
    Gmat = [gR(:), gG(:), gB(:)];          % 300×3
    den  = [sum(gR); sum(gG); sum(gB)];    % 3×1

    % ---- 数据拟合项（相对误差）----
    expo  = -(gH*cH + gE*cE);              % 300×Ns
    ODhat = -log10( (Gmat.' * 10.^(expo)) ./ den );  % 3×Ns

    eps_   = 1e-6;
    
    % ======= H R段惩罚因子
     % 通道权重: R=1, G=1, B=0.5
    w = [1; 1; 1];      

    diff   = (ODhat - P).^2;      % 3×Ns
    diff_w = w .* diff;           % 加权 3×Ns

    num = sqrt(sum(diff_w, 1));   % 1×Ns
    denom = max(eps_, sqrt(sum(P.^2,1)));

    relErr = num./ denom ;        % 1×Ns


    % relErr = sqrt(sum((ODhat - P).^2,1)) ./ max(eps_, sqrt(sum(P.^2,1)));
    Res    = mean(relErr);
    core   = Res/(1+Res);                  % 压缩到 (0,1)，减弱 outlier 影响
   
    % ======= OD space point-to-point distance =======
    % % 通道权重（等价于加权欧氏度量）
    % w = [1; 1; 1];      % 3×1
    % W = diag(w);        % 3×3
    % 
    % D = ODhat - P;      % 3×Ns
    % 
    % % 每个样本在 OD 空间中的加权欧氏距离
    % dist_i = sqrt(sum((W * D) .* D, 1));   % 1×Ns
    % 
    % % core：空间中两点集合的平均距离
    % core = mean(dist_i);


    % ============================================================
    % 1) 光谱正交 & 平滑（可选）
    % ============================================================
    cos_he = (gH(:)'*gE(:)) / max(1e-8, norm(gH)*norm(gE));
    D2     = convmtx([1 -2 1], numel(gH)); 
    D2     = D2(1:numel(gH),1:numel(gH));
    smoothH = norm(D2*gH,2)^2; 
    smoothE = norm(D2*gE,2)^2;

    R_ortho_smooth = lambda_ortho * cos_he^2 ;%+ lambda_s*(smoothH + smoothE)

    % ============================================================
    % 2) 模板正则: 让 H/E 光谱靠近参考光谱
    %     R_temp = λ_temp ( ||gH - gH_ref||^2 + ||gE - gE_ref||^2 )
    % ============================================================
    if nargin >= 10 && ~isempty(gH_ref) && ~isempty(gE_ref) ...
            && lambda_temp > 0
        % 确保形状一致
        if numel(gH_ref) ~= numel(gH) || numel(gE_ref) ~= numel(gE)
            error('Template spectra size mismatch with gH/gE.');
        end

        %  L = numel(gH);
        % % 波长轴：假设光谱对应 400~700nm 等间距
        % % wl_axis = linspace(1,300,L).';
        % 
        % % ---- H 的权重：红尾 620–700nm 权重大一些 ----
        % wH = ones(L,1);
        % % w_red_factor = 5;               % 红段差异放大倍数，可调 3~10
        % % red_mask = (wl_axis >= 220) & (wl_axis <= 300);
        % wH(220:end) = 5;
        % 
        % % 带权差异
        % diffH = wH .* (gH - gH_ref(:)); % H: 红尾差异更“贵”
       
        diffH = gH - gH_ref(:);
        diffE = gE - gE_ref(:);
        temp = norm(diffH,2)^2 + norm(diffE,2)^2;
        temp=temp/(1+temp);     
        R_temp = lambda_temp * temp;
    else
        R_temp = 0;
    end
    % ============================================================
    % 3) Channel-wise fractional RGB boundary regularization
    % ============================================================
    L = numel(gR);

    n_edge = 20;
    I_low  = 1:n_edge;
    I_high = (L-n_edge+1):L;
    eps_q = 1e-12;

    eta_R_low = sum(gR(I_low).^2) / (sum(gR.^2) + eps_q);
    eta_G_low = sum(gG(I_low).^2) / (sum(gG.^2) + eps_q);
    eta_G_high = sum(gG(I_high).^2) / (sum(gG.^2) + eps_q);
    eta_B_high = sum(gB(I_high).^2) / (sum(gB.^2) + eps_q);

    E_RGB = eta_R_low + eta_G_low + eta_G_high + eta_B_high;
    rho_RGB = E_RGB / (1 + E_RGB);

    lambda_rgb_edge = 1e-3;
    R_rgb_edge = lambda_rgb_edge * rho_RGB;

    S = core + R_temp + R_rgb_edge;
end

% function S = local_cost_with_fixed_c_reg(p_all, P, cH, cE, g5, lambda_ortho, lambda_s)
%     idx = @(k) (15*(k-1)+1):(15*k);
%     pH = p_all(idx(1));  pE = p_all(idx(2));
%     pR = p_all(idx(3));  pG = p_all(idx(4));  pB = p_all(idx(5));
%     gH = g5(pH); gE = g5(pE);
%     gR = g5(pR); gG = g5(pG); gB = g5(pB);
%     Gmat=[gR(:),gG(:),gB(:)];
%     den =[sum(gR);sum(gG);sum(gB)];
%     expo=-(gH*cH + gE*cE);
%     ODhat = -log10( (Gmat.' * 10.^(expo)) ./ den );
%     eps_ = 1e-6;
%     relErr = sqrt(sum((ODhat - P).^2,1)) ./ max(eps_, sqrt(sum(P.^2,1)));
%     Res = mean(relErr);
%     core = Res/(1+Res);
% 
%     % 正则：正交 + 二阶平滑（H/E）
%     cos_he = (gH(:)'*gE(:)) / max(1e-8, norm(gH)*norm(gE));
%     D2 = convmtx([1 -2 1], numel(gH)); D2=D2(1:numel(gH),1:numel(gH));
%     smoothH = norm(D2*gH,2)^2; smoothE = norm(D2*gE,2)^2;
% 
%     % S = core + lambda_ortho * cos_he^2 + lambda_s*(smoothH + smoothE);
%     % ---- 颜色先验：让 E 偏G、少B；H 偏B ----
%     % 用内积近似“通道吸收打分”
%     sE = [dot(gR,gE), dot(gG,gE), dot(gB,gE)];  % E 对 R/G/B
%     sH = [dot(gR,gH), dot(gG,gH), dot(gB,gH)];  % H 对 R/G/B
%     relu = @(x) max(0,x);
% 
%     % E: 希望 G 最大、B 最小
%     penE = relu(sE(3) - sE(2)) + relu(sE(1) - sE(2));  % 惩罚 B>G 和 适度惩罚 R>G
%     % % H: 希望 B 最大
%     % penH = relu(sH(2) - sH(3)) + relu(sH(1) - sH(3));
% 
%     % 额外：直接压 gE 的“蓝能量”
%     penE_blue_area = mean(gB.^2 .* gE.^2);   % 或 dot(gB,gE)^2 也行
% 
%     lambda_color = 1e-2;          % 起步 1e-3~1e-2
%     lambda_eblue = 1e-4;          % 压蓝辅助项，起步 1e-4~1e-3
%     color_prior = lambda_color*(penE);% + lambda_eblue*penE_blue_area;
% 
%     S = core ;%+ lambda_ortho*cos_he^2 + lambda_s*(smoothH+smoothE);+color_prior
% 
% end

% function [c,ceq] = nonlcon_rgb_peak_le1(p_all)
%     idx = @(k) (15*(k-1)+1):(15*k);
%     g   = @(p) Gaussian_reconstruction5(p);
%     pR = p_all(idx(3));  pG = p_all(idx(4));  pB = p_all(idx(5));
%     gR = g(pR); gG=g(pG); gB=g(pB);
%     c   = [max(gR)-1; max(gG)-1; max(gB)-1];
%     ceq = [];
% end
% 
% function ofun = stopAtIterFactory(maxIter)
%     ofun = @(x,optimValues,state) stopAtIter(x,optimValues,state,maxIter);
% end
% 
% function stop = stopAtIter(~,optimValues,state,maxIter)
%     stop = false;
%     if strcmp(state,'init') && maxIter <= 0
%         stop = true; return;
%     end
%     if strcmp(state,'iter') && optimValues.iteration >= maxIter
%         stop = true;
%     end
% end

function [P, P_test, N, idx_samp, idx_samp_test, meta] = samplePixelsNonwhite(Ibig, Ibig_test, numSample, thr, seed, doPlot, lowCut)
    if nargin < 7 || isempty(lowCut), lowCut = 15; end
    if nargin < 6 || isempty(doPlot), doPlot = true; end
    if nargin < 5, seed = 42; end
    if nargin < 4 || isempty(thr), thr = 235; end
    if nargin < 3 || isempty(numSample), numSample = 200; end

    I_rgb      = double(Ibig);
    I_rgb_test = double(Ibig_test);
    [m_full, n_full, ~]           = size(I_rgb);
    [m_full_test, n_full_test, ~] = size(I_rgb_test);

    I_od_full      = rgb2od(I_rgb);
    I_od_full_test = rgb2od(I_rgb_test);

    P_all      = reshape(I_od_full,      [], 3)';   % 3×N_all
    P_all_test = reshape(I_od_full_test, [], 3)';   % 3×N_all_test

    I_u8      = uint8(Ibig);
    I_u8_test = uint8(Ibig_test);
    max_train = max(I_u8, [], 3);  min_train = min(I_u8, [], 3);
    max_test  = max(I_u8_test, [], 3); min_test  = min(I_u8_test, [], 3);

    mask_train = (max_train <= thr) & (min_train >= lowCut);
    mask_test  = (max_test  <= thr) & (min_test  >= lowCut);
    pool_train = find(mask_train); pool_test = find(mask_test);

    if isempty(pool_train), error('训练图像没有符合条件的像素（thr=%d, lowCut=%d）。', thr, lowCut); end
    if isempty(pool_test),  error('测试图像没有符合条件的像素（thr=%d, lowCut=%d）。', thr, lowCut); end

    if ~isempty(seed), rng(seed); end
    take_train = min(numSample, numel(pool_train));
    take_test  = min(numSample, numel(pool_test));
    idx_samp      = pool_train(randperm(numel(pool_train), take_train));
    idx_samp_test = pool_test (randperm(numel(pool_test),  take_test));

    P      = P_all(:, idx_samp);
    P_test = P_all_test(:, idx_samp_test);
    N      = size(P, 2);

    [y_all, x_all]           = ind2sub([m_full,       n_full],       idx_samp);
    [y_all_test, x_all_test] = ind2sub([m_full_test, n_full_test],   idx_samp_test);

    meta = struct();
    meta.m_full = m_full; meta.n_full = n_full;
    meta.m_full_test = m_full_test; meta.n_full_test = n_full_test;
    meta.x_all = x_all; meta.y_all = y_all;
    meta.x_all_test = x_all_test; meta.y_all_test = y_all_test;
    meta.thr = thr; meta.lowCut = lowCut;

    if doPlot
        figure('Name','Random Sample Locations (Non-white & Non-black only)');
        subplot(1,2,1);
        imshow(uint8(I_rgb)); hold on; plot(x_all, y_all, 'r.', 'MarkerSize', 8);
        title(sprintf('Train (%d / pool %d) thr=%d, lowCut=%d', N, numel(pool_train), thr, lowCut));
        subplot(1,2,2);
        imshow(uint8(I_rgb_test)); hold on; plot(x_all_test, y_all_test, 'g.', 'MarkerSize', 8);
        title(sprintf('Test  (%d / pool %d) thr=%d, lowCut=%d', size(P_test,2), numel(pool_test), thr, lowCut));
    end
end
function [cH,cE,zH,zE,uH,uE,Hst] = solve_C_ADMM(P,gH,gE,Gmat,den,...
    cH,cE,zH,zE,uH,uE,optsPixJac,rho,lambda_c, admmIters,lambda_x, ...
    tau, cH0, cE0, wHX, wEX )

    Ns = size(P,2);
    Hst.r_pri  = zeros(1,admmIters);
    Hst.r_dual = zeros(1,admmIters);
    Hst.Jc     = zeros(1,admmIters);
    zH_prev = zH; zE_prev = zE;

    
    for it = 1:admmIters
        % ---- (1) c-update：并行最小二乘 ----
        vH = zH - uH; vE = zE - uE;
        parfor j = 1:Ns
            Pj = P(:,j);
            vj = [vH(j); vE(j)];
            x_prev = [cH0(j); cE0(j)];    % MM“接触点”，在本轮外层固定
            fun    = @(x) pix_residual_aug_with_jac(x,Pj,gH,gE,Gmat,den,rho,vj, tau, x_prev);
            x0     = [cH(j); cE(j)];
            x = lsqnonlin(fun, x0, [-0.05;-0.05], [10;10], optsPixJac);
            cH(j)=x(1); cE(j)=x(2);
        end
    
        tH = cH + uH; 
        tE = cE + uE;
        
        % 若需要非负约束，保留 max(·,0)；若允许负值，可去掉
        % tH = max(tH, 0);
        % tE = max(tE, 0);
        
        % ---- 参数 ----
        % lambda_c: 稀疏正则系数（可调，控制稀疏强度）
        % rho:      ADMM penalty（一般 1~10）
        % tauH = lambda_c / rho;     % H 通道的软阈值
        % tauE = lambda_c / rho;     % E 通道的软阈值
        
        % 若希望 H 比 E 更稀疏（防止H吞E），可稍微放大 H 的阈值，例如：
        tauH = 1.5 * lambda_c / rho;
        tauE = 1.0 * lambda_c / rho;
        
        % % ---- Soft-threshold (逐元素独立 L1) ----
        % zH = sign(tH) .* max(abs(tH) - tauH, 0);
        % zE = sign(tE) .* max(abs(tE) - tauE, 0);

        % 互斥额外阈值（与对方幅值成比例）
        alpha = lambda_x / rho;        % 互斥 → 等效阈值系数
        % 第一次：用 tE/tH 估计
        tauHX = alpha * abs(tE);
        tauEX = alpha * abs(tH);

        % 第一次软阈值
        zH = soft_threshold(tH, tauH + tauHX);
        zE = soft_threshold(tE, tauE + tauEX);

        % % 可选：做 1~2 次小的固定点细化（用 z 的新值更新互斥阈值）
        % for kk = 1:1   % 如需更强互斥可设为 2
        %     tauHX = alpha * abs(zE);
        %     tauEX = alpha * abs(zH);
        %     zH = soft_threshold(tH, tauH + tauHX);
        %     zE = soft_threshold(tE, tauE + tauEX);
        % end

        % ===== 使用“外层冻结”的互斥权重（固定） =====
        % wHX, wEX 已经作为函数入参传进来，与 tH,tE 无关，不在内层更新
        % zH = soft_threshold(tH, tauH + wHX);
        % zE = soft_threshold(tE, tauE + wEX);


        
        % ---- 非负约束（如果物理上不允许负浓度） ----
        zH = max(zH, 0);
        zE = max(zE, 0);

        % ---- (3) u-update ----
        uH = uH + (cH - zH);
        uE = uE + (cE - zE);
        % ---- (4) 记录残差与子目标 ----
        Hst.r_pri(it)  = norm(cH - zH) + norm(cE - zE);
        Hst.r_dual(it) = rho*( norm(zH - zH_prev) + norm(zE - zE_prev) );
        zH_prev = zH; zE_prev = zE;

        [Jfid_tmp, L1_tmp] = datafid_and_l1_fixedZ(P, gH, gE, Gmat, den, cH, cE, lambda_c, zH, zE);
        Hst.Jc(it) = Jfid_tmp + L1_tmp;

        % 记录 ρ & 阈值
        Hst.rho_hist(it)  = rho;
        Hst.tauH_hist(it) = tauH;
        Hst.tauE_hist(it) = tauE;
        
        % ADMM判停
        N = numel(cH);
       eps_abs = 1e-5; eps_rel = 5e-4;
        r_pri  = norm(cH - zH) + norm(cE - zE);
        r_dual = rho*( norm(zH - zH_prev) + norm(zE - zE_prev) );
        
        thr_pri  = eps_abs*sqrt(2*N) + eps_rel*max(norm([cH cE]), norm([zH zE]));
        thr_dual = eps_abs*sqrt(2*N) + eps_rel*rho*norm([uH uE]);
        
        if (r_pri <= thr_pri) && (r_dual <= thr_dual)
            break;   % ADMM 内层收敛
        end

    end

end


function [r, J] = pix_residual_aug_with_jac(x, Pj, gH, gE, Gmat, den, rho, v, tau, x_prev)

% 与你原来的 pix_residual_with_jac 相同，但把 ADMM 二次项拼进去

    den  = max(den, 1e-12);

    S  = gH*x(1) + gE*x(2);      % 300x1
    Y  = 10.^(-S);                % 300x1
    T  = Gmat.' * Y;             % 3x1
    T  = max(T, 1e-12);
    F  = -log10( T ./ den );     % 3x1
    r_data = F - Pj;

    SH = Gmat.' * (gH .* Y);     % 3x1
    SE = Gmat.' * (gE .* Y);     % 3x1
    JH = SH ./ T;
    JE = SE ./ T;
    J_data = [JH, JE];           % 3x2

    % ADMM 二次罚项
    r_quad = sqrt(rho) * (x - v);           % 2x1
    J_quad = sqrt(rho) * eye(2);            % 2x2

    % 拼接
    % === MM 稳定化（主要化上界；在当前点接触且处处上界） ===
    r_stab = sqrt(tau) * (x - x_prev);   % 2x1
    J_stab = sqrt(tau) * eye(2);         % 2x2
    
    % 拼接（数据残差 + ADMM 二次 + MM 稳定项）
    r = [r_data; r_quad; r_stab];
    J = [J_data; J_quad; J_stab];

end
function [c,ceq] = nonlcon_identity_rgb_peak(p_all)
    idx = @(k) (15*(k-1)+1):(15*k);
    g   = @(p) Gaussian_reconstruction5(p);
    pH=p_all(idx(1)); pE=p_all(idx(2));
    pR=p_all(idx(3)); pG=p_all(idx(4)); pB=p_all(idx(5));
    gH=g(pH); gE=g(pE); gR=g(pR); gG=g(pG); gB=g(pB);

    % 光谱质心近似峰位
    w=(1:numel(gH)).';
    centH=(w'*(gH.^2))/sum(gH.^2);
    centE=(w'*(gE.^2))/sum(gE.^2);
    mLam=0;     % 最小间隔（索引单位）
    c1=(centE+mLam)-centH;  % ≤0 表示H在更高波长
    cRGB=[max(gR)-1; max(gG)-1; max(gB)-1];
    c=[c1; cRGB];
    ceq=[];
end
function [c, ceq] = nonlcon_physical_HE_relation(p_all)
% H/E label-identification constraint: lambda_bar(E) <= lambda_bar(H).

    idxBlk = @(k) (15*(k-1)+1):(15*k);
    g5 = @(p) Gaussian_reconstruction5(p);

    pH = p_all(idxBlk(1));
    pE = p_all(idxBlk(2));

    gH = g5(pH); gH = gH(:);
    gE = g5(pE); gE = gE(:);

    L = numel(gH);
    lambda_axis = linspace(400, 700, L)';

    centH = sum(lambda_axis .* gH.^2) / (sum(gH.^2) + 1e-12);
    centE = sum(lambda_axis .* gE.^2) / (sum(gE.^2) + 1e-12);

    % fmincon uses c <= 0.
    c = centE - centH;
    ceq = [];
end

function mu_nm = map_mu_to_nm(mu_vec, wl_axis, L)
    % 若 μ 是索引（1..L），映射到 nm；若你明确 μ 已是 nm，可直接 return mu_vec。
    if all(mu_vec >= 1-1e-6 & mu_vec <= L+1e-6)
        % 视为索引
        mu_idx = round(mu_vec);
        mu_idx = min(max(mu_idx,1), L);
        mu_nm  = wl_axis(mu_idx);
    else
        % 看作本就是 nm
        mu_nm = mu_vec;
    end
end


function z_all = proj_l1_ball_nonneg(v, kappa)
    % 投影到非负 L1-ball:  min_z 0.5*||z-v||^2 s.t. z>=0, sum(z)<=kappa
    v = max(v, 0);
    if sum(v) <= kappa
        z_all = v; return;
    end
    % 找阈值 theta
    s = sort(v, 'descend');
    cssv = cumsum(s);
    rho = find(s > (cssv - kappa) ./ (1:numel(s))', 1, 'last');
    theta = (cssv(rho) - kappa) / rho;
    z_all = max(v - theta, 0);
end

function y = soft_threshold(x, tau)
    % tau 可为标量或与 x 同长向量
    y = sign(x) .* max(abs(x) - tau, 0);
end

function [fid_mean, l1sum] = datafid_and_l1(P, gH, gE, Gmat, den, cH, cE, lambda_c)
    eps_ = 1e-6;
    expo = -(gH*cH + gE*cE);                 % 300 x Ns
    ODhat = -log10( (Gmat.' * 10.^(expo)) ./ den );  % 3 x Ns
    relErr = sqrt(sum((ODhat - P).^2,1)) ;%./ max(eps_, sqrt(sum(P.^2,1)))
    fid_mean = mean(relErr);
    l1sum = lambda_c * sum(abs(cH) + abs(cE));
end

function ofun = stopAtIterFactory(maxIter)
% 返回一个 OutputFcn：在 init 就截停（当 maxIter<=0），或当迭代数达到 maxIter 时截停
    ofun = @(x,optimValues,state) stopAtIter(x,optimValues,state,maxIter);
end

function stop = stopAtIter(~,optimValues,state,maxIter)
    stop = false;
    % —— 0 次迭代：在真正进入第一步前就停（最贴近“初始点”）
    if strcmp(state,'init') && maxIter <= 0
        stop = true;
        return;
    end
    % —— k 次迭代：当迭代序号达到 k 就停
    % 说明：optimValues.iteration 通常从 0 开始，第一次迭代后为 1
    if strcmp(state,'iter') && optimValues.iteration >= maxIter
        stop = true;
    end
end
function [cos2_val, smoothH, smoothE] = reg_terms(gH,gE)
    cos_he = (gH(:)'*gE(:)) / max(1e-8, norm(gH)*norm(gE));
    cos2_val = cos_he^2;
    D2 = convmtx([1 -2 1], numel(gH)); D2=D2(1:numel(gH),1:numel(gH));
    smoothH = norm(D2*gH,2)^2; smoothE = norm(D2*gE,2)^2;
end
%% ====== 新增的目标计算辅助 ======


function [Jfid, L1z] = datafid_and_l1_fixedZ(P, gH, gE, Gmat, den, cH, cE, lambda_c, zH, zE)
    % C步子目标：用 cH/cE 计算数据项，用 zH/zE 计算 L1（ADMM习惯）
    [Jfid, ~] = datafid_and_l1(P, gH, gE, Gmat, den, cH, cE, 0);
    L1z  = lambda_c * sum(abs(zH) + abs(zE));
end

function [gH,gE,gR,gG,gB,Gmat,den] = get_gs_from_p(p_all,gfun)
    idx = @(k) (15*(k-1)+1):(15*k);
    gH = gfun(p_all(idx(1)));  
    gE = gfun(p_all(idx(2)));
    gR = gfun(p_all(idx(3)));  
    gG = gfun(p_all(idx(4)));  
    gB = gfun(p_all(idx(5)));
    Gmat=[gR(:),gG(:),gB(:)];
    den =[sum(gR);sum(gG);sum(gB)];
end

function [p0_best, score_all] = choose_multistart_p0( ...
            p0_base, P_s, lb, ub, ...
            lambda_ortho, lambda_s, lambda_c, lambda_x, ...
            g5, ...
            idxBlk, Aidx, ...
            maxOuter_small, wantIters_p_small, admmIters_small, ...
            baseOpt, optsPixJac,K_start)
% =========================================================================
% 在 P 上做 multi-start:
%   - 以 p0_base 为中心，生成 K 个扰动起点 p0^(r)
%   - 对每个起点跑一个“小 BCD”（C-step + P-step）
%   - 用 data term + 简单物理打分 做评分
%   - 返回评分最小的 p0_best
% =========================================================================


    rng(2025);        % 固定随机种子，保证 multi-start 可复现

    % --- 生成多组扰动起点 -------------------------------------------
    p0_list  = cell(K_start,1);
    scores   = zeros(K_start,1);

    for r = 1:K_start
        if r == 1
            % 第一个起点直接用原始 p0_base
            p0_list{r} = p0_base;
        else
            % 后面的起点在原始附近做扰动
            p0_list{r} = perturb_p0_gaussian(p0_base, idxBlk, Aidx, lb, ub);
        end
    end

    % --- 准备“小 BCD”里的优化选项 -------------------------------
    % P-step 非精确迭代次数
    optIterP_small = optimoptions(baseOpt, ...
        'OutputFcn', stopAtIterFactory(wantIters_p_small), ...
        'Display','off');

    % C-step 的 ADMM 罚参数和稳定化参数可以取你大 BCD 里的典型值
    rho_small  = 4.0;   % 较小的 ρ，一般足够
    tau0_small = 1e-1;  % MM 稳定化

    % --- 对每个起点跑小 BCD，并打分 ------------------------------
    for r = 1:K_start
        p_cur = p0_list{r};

        % 初始 C / Z / U
        [gH,gE,gR,gG,gB,Gmat,den] = get_gs_from_p(p_cur, g5);
        Ns = size(P_s,2);
        cH = zeros(1,Ns);
        cE = zeros(1,Ns);
        zH = cH; zE = cE;
        uH = zeros(1,Ns); uE = zeros(1,Ns);

        % 小 BCD 循环
        for it = 1:maxOuter_small
            % --- (A) C-step: ADMM+MM 小迭代 ---
            alpha = lambda_x / rho_small;
            wHX = alpha * abs(zE);   % 互斥权重 frozen
            wEX = alpha * abs(zH);

            [cH,cE,zH,zE,uH,uE,~] = solve_C_ADMM( ...
                P_s, gH, gE, Gmat, den, ...
                cH, cE, zH, zE, uH, uE, ...
                optsPixJac, rho_small, lambda_c, ...
                admmIters_small, lambda_x, ...
                tau0_small, cH, cE, wHX, wEX);

            % --- (B) P-step: 简化 prox-MM（不做 backtracking，只用固定 gamma_p_small） ---
            gamma_p_small = 1e-3;   % 小 BCD 里固定一个稍微保守的 proximal 强度

            F_core = @(p_all) local_cost_with_fixed_c_reg( ...
                                p_all, P_s, zH, zE, g5, ...
                                lambda_ortho, lambda_s, ...
                                0, [], []); % 这里小 BCD 不用 template 项，可简化

            costP_small = @(p_all) F_core(p_all) ...
                            + 0.5 * gamma_p_small * sum((p_all - p_cur).^2);

            % 非精确 P-step
            [p_new, ~] = fmincon(costP_small, p_cur, [], [], [], [], ...
                                 lb, ub, @nonlcon_physical_HE_relation, ...
                                 optIterP_small);
            p_cur = p_new;

            % 更新光谱（供下一轮 C-step 使用）
            [gH,gE,gR,gG,gB,Gmat,den] = get_gs_from_p(p_cur, g5);
        end

        % --- 对第 r 个起点的最终结果打分 -------------------------
        % 评分1：data fidelity（使用最终 p_cur, cH,cE）
        [Jfid, ~] = datafid_and_l1(P_s, gH, gE, Gmat, den, cH, cE, 0);

        % 评分2：简单物理先验惩罚（H/E 的质心顺序 + 相关性）
        score_phys = physical_score_HE(gH, gE);

        scores(r) = Jfid ;  % 物理惩罚权重 1e-2 可调     + 1e-2 * score_phys

        fprintf('  [Multi-start] r=%d, Jfid=%.3e, phys=%.3e, total=%.3e\n', ...
                r, Jfid, score_phys, scores(r));
    end

    [~, best_idx] = min(scores);
    p0_best = p0_list{best_idx};

    % 输出一些信息，便于调试/画图
    score_all = struct();
    score_all.values   = scores;
    score_all.best_idx = best_idx;
end
function p_pert = perturb_p0_gaussian(p0_base, idxBlk, Aidx, lb, ub)
% 对 p0_base 做一个小扰动（在 box 约束内）:
%   - 幅度 A: 乘以 0.9~1.1 的随机因子
%   - 均值 mu: 加减一个小偏移
%   - sigma: 加减一个小偏移
    p_pert = p0_base;

    % 每个通道单独扰动
    for k = 1:5
        bi = idxBlk(k);
        pk = p_pert(bi);

        % A, mu, sigma 的索引
        Aidx_loc = Aidx;
        muidx_loc = Aidx + 1;
        sidx_loc  = Aidx + 2;

        % 幅度扰动
        for ii = 1:numel(Aidx_loc)
            ai = Aidx_loc(ii);
            pk(ai) = pk(ai) * (0.9 + 0.2*rand());    % [0.9,1.1]
        end

        % mu 扰动（±5 索引单位）
        for ii = 1:numel(muidx_loc)
            mi = muidx_loc(ii);
            pk(mi) = pk(mi) + (rand()-0.5)*10;       % [-5,5]
        end

        % sigma 扰动（±10%）
        for ii = 1:numel(sidx_loc)
            si = sidx_loc(ii);
            pk(si) = pk(si) * (0.9 + 0.2*rand());
        end

        p_pert(bi) = pk;
    end

    % 投影回 box 约束
    p_pert = project_to_open_box(p_pert, lb, ub);
end
function s = physical_score_HE(gH, gE)
% 一个非常简单的 H/E 物理打分:
%   - 惩罚 H/E 相似（内积大）
%   - 惩罚 E 在 B 段能量过大
%   - 惩罚 E 在 G 段不够强

    gH = gH(:); gE = gE(:);
    L  = numel(gH);
    w  = linspace(1,300,L)';

    % 分段：B:1-100, G:101-200, R:201-300
    Bband = 1:round(L/3);
    Gband = (round(L/3)+1):round(2*L/3);
    Rband = (round(2*L/3)+1):L;

    % 1) 相似度惩罚（越相似越大）
    cos_he = (gH'*gE) / max(1e-8, norm(gH)*norm(gE));
    pen_corr = cos_he^2;

    % 2) E 的分段能量
    areaB_E = sum(gE(Bband));
    areaG_E = sum(gE(Gband));
    areaR_E = sum(gE(Rband));

    % 惩罚 E 在 B 段太强（希望 E 更偏 G/R）
    pen_EB = max(0, areaB_E - 0.5*areaG_E);  % B 不应超过 0.5*G

    % 惩罚 G 段太弱
    pen_EG = max(0, 0.5*areaR_E - areaG_E);  % G 至少大于 0.5*R

    s = pen_corr + pen_EB + pen_EG;
end
function [cH, cE] = solve_C_smooth(P, gH, gE, Gmat, den, ...
                                      cH_init, cE_init, ...
                                      optsPixJac, lambda2)
% ------------------------------------------------------------
% C-step：带 L2 正则的非线性最小二乘（逐像素）
%
% 目标：min_{cH,cE>=0}  ||F(c) - P||^2 + lambda2 * ||c||^2
%
% P       : 3×Ns  采样 OD
% gH,gE   : 300×1 染料光谱
% Gmat    : 300×3 RGB 相机响应矩阵
% den     : 3×1   归一化因子
% cH_init : 1×Ns  上一轮的初值
% cE_init : 1×Ns
% lambda2 : L2 正则项权重（建议 1e-4 ~ 1e-2）
% ------------------------------------------------------------

    Ns  = size(P, 2);
    cH  = cH_init;
    cE  = cE_init;

    den = max(den, 1e-12);

    parfor j = 1:Ns
        Pj  = P(:, j);
        x0  = [cH_init(j); cE_init(j)];   % 以上一轮解为初值

        fun = @(x) pix_residual_L2_with_jac( ...
                        x, Pj, gH, gE, Gmat, den, lambda2);

        % C>=0 的非负约束
        x = lsqnonlin(fun, x0, [-0.05;-0.05], [10;10], optsPixJac);
        cH(j) = x(1);
        cE(j) = x(2);
    end
end
function [r, J] = pix_residual_L2_with_jac(x, Pj, gH, gE, Gmat, den, lambda2)
% ------------------------------------------------------------
% 像素级残差（带 L2 正则）的 Jacobian 版本：
%
% r = [ r_data ;
%       r_reg  ]
%
% 其中：
%   r_data = F(x) - Pj     (3x1)
%   r_reg  = sqrt(lambda2) * x   (2x1)
%
% 所以等价目标是：
%   ||F(x) - Pj||^2 + lambda2 * ||x||^2
% ------------------------------------------------------------

    % ===== 数据项（和你原来的 pix_residual_with_jac 相同）=====
    den  = max(den, 1e-12);

    S  = gH*x(1) + gE*x(2);      % 300×1
    Y  = 10.^(-S);               % 300×1
    T  = Gmat.' * Y;             % 3×1
    T  = max(T, 1e-12);
    F  = -log10( T ./ den );     % 3×1

    r_data = F - Pj;

    SH = Gmat.' * (gH .* Y);     % 3×1
    SE = Gmat.' * (gE .* Y);     % 3×1
    JH = SH ./ T;
    JE = SE ./ T;
    J_data = [JH, JE];           % 3×2

    % ===== L2 正则项 =====
    if lambda2 > 0
        s     = sqrt(lambda2);
        r_reg = s * x;           % 2×1
        J_reg = s * eye(2);      % 2×2

        % 拼接：总残差 & 总雅可比
        r = [r_data; r_reg];     % 5×1
        J = [J_data; J_reg];     % 5×2
    else
        % lambda2 = 0 时退化为原始无正则形式
        r = r_data;
        J = J_data;
    end
end


function [gH_ref, gE_ref, gR_ref, gG_ref, gB_ref, gMat_ref,den_ref] = ...
    estimate_ref_spectra_from_image(refImagePath, numSample, thrWhite, lowCut, ...
                                    g5, idxBlk, Aidx, baseOpt, optsPixJac, ...
                                    lambda_ortho, lambda_s, lambda_temp)
% =========================================================================
% 从一张参考图像中估计 H/E/R/G/B 光谱，用的是简化版 BCD
% 只跑一次，不做输出图像，只要得到 p_opt -> gH/gE/gRGB
% =========================================================================

    fprintf('[REF] Start estimating spectra from reference image...\n');

    Iref = imread(refImagePath);
    I_rgb = double(Iref);
    I_od  = rgb2od(I_rgb);
    [m,n,~] = size(Iref);
    P_full = reshape(I_od, [], 3)';   % 3 x N
    N = m*n;

    % —— 采样像素（跟主流程一样）——
    [P_s, ~, Ns, ~, ~, ~] = samplePixelsNonwhite(Iref, Iref, numSample, thrWhite, 42, false, lowCut);

    % —— 初始 p0 和 box 约束（与主流程一致）——
    load p_fit5_result_H_global.mat p_fit; pH0 = p_fit(:);
    load p_fit5_result_E_global.mat p_fit; pE0 = p_fit(:);
    load p_fit5_result_R_global.mat p_fit; pR0 = p_fit(:);
    load p_fit5_result_G_global.mat p_fit; pG0 = p_fit(:);
    load p_fit5_result_B_global.mat p_fit; pB0 = p_fit(:);
    p0 = [pH0; pE0; pR0; pG0; pB0];

    lb = -inf(size(p0));
    ub =  inf(size(p0));
    muidx = [2 5 8 11 14];
    sidx  = [3 6 9 12 15];

    for kk = 1:5
        bi = idxBlk(kk);
        lb(bi(Aidx)) = 0;
        ub(bi(Aidx)) = 1;
        lb(bi(sidx)) = 1e-3;
    end

    bR = idxBlk(3);
    bG = idxBlk(4);
    bB = idxBlk(5);

    lb(bR(sidx)) = 10;    ub(bR(sidx)) = 80;
    lb(bG(sidx)) = 10;    ub(bG(sidx)) = 80;
    lb(bB(sidx)) = 10;    ub(bB(sidx)) = 80;

    lb(bB(muidx)) = 20;     ub(bB(muidx)) = 130;
    lb(bG(muidx)) = 90;     ub(bG(muidx)) = 210;
    lb(bR(muidx)) = 150;    ub(bR(muidx)) = 290;

    p_cur = project_to_open_box(p0, lb, ub);

    % —— 初始化 C —— 
    pH=p_cur(1:15); pE=p_cur(16:30);
    pR=p_cur(31:45); pG=p_cur(46:60); pB=p_cur(61:75);
    gH=g5(pH); gE=g5(pE); gR=g5(pR); gG=g5(pG); gB=g5(pB);
    gH_ini = gH;gE_ini = gE;
    Gmat=[gR(:),gG(:),gB(:)]; den=[sum(gR);sum(gG);sum(gB)];
    cH=zeros(1,Ns); cE=zeros(1,Ns);

    % —— BCD 参数（参考图可以比主流程略少一点）——
    maxOuter   = 60;
    minOuter   = 5;
    tol_relP   = 5e-4;
    tol_relC   = 5e-4;
    wantIters_p = 2;
    L_P_current = 1e-3;

    % alpha_hist   = 1e-1;   % 强度（建议：1e-2 ~ 1e-1）
    % alpha_update = 0.5;     % 最佳：0.2 ~ 0.5

    alpha_hist   = 1e-3;   % 强度（建议：1e-2 ~ 1e-1）
    alpha_update = 0.8;     % 最佳：0.2 ~ 0.5


    optIterP = optimoptions(baseOpt,'OutputFcn',stopAtIterFactory(wantIters_p), ...
                                      'Display','off');

    p_k_minus_1  = p_cur;
    cH_k_minus_1 = cH;
    cE_k_minus_1 = cE;
    % 初始化历史浓度（非常重要）
    cH_hist = zeros(size(cH));
    cE_hist = zeros(size(cE));


    for it = 1:maxOuter
        % --- C-step：用当前 p_k_minus_1 解浓度（平滑版，无 L1）---
        [gH,gE,gR,gG,gB,Gmat,den] = get_gs_from_p(p_k_minus_1,g5);
        [cH, cE, cH_hist, cE_hist] = solve_C_smooth_history( ...
                                    P_s, gH, gE, Gmat, den, ...
                                    cH, cE, ...
                                    cH_hist, cE_hist, ...
                                    optsPixJac, ...
                                    alpha_hist, alpha_update );

        % --- P-step：prox-MM（不加模板约束或用很小 lambda_temp）---
        F_core = @(p_all) local_cost_with_fixed_c_reg( ...
                            p_all, P_s, cH, cE, g5, ...
                            lambda_ortho, lambda_s, lambda_temp, ...
                            gH_ini, gE_ini);  % 这里给模板光谱

        L_try = L_P_current;
        maxBT = 5;
        bt_cnt = 0;
        F_old = F_core(p_k_minus_1);

        while true
            gamma_p = L_try;
            costP = @(p_all) F_core(p_all) ...
                      + 0.5 * gamma_p * sum((p_all - p_k_minus_1).^2);

            [p_trial, ~] = fmincon(costP, p_k_minus_1, [], [], [], [], ...
                                   lb, ub, @nonlcon_physical_HE_RGB_relation, optIterP);

            F_new = F_core(p_trial);
            dP    = p_trial - p_k_minus_1;
            rhs   = F_old + 0.5 * L_try * sum(dP.^2);

            if F_new <= rhs
                p_cur       = p_trial;
                L_P_current = L_try;
                break;
            else
                L_try = L_try * 5;
                bt_cnt = bt_cnt + 1;
                if bt_cnt >= maxBT
                    warning('[REF] backtracking reached maxBT; accept current p_trial.');
                    p_cur       = p_trial;
                    L_P_current = L_try;
                    break;
                end
            end
        end
        [p_cur, cH, cE, cH_hist, cE_hist] = normalize_HE_RGB_state(p_cur, cH, cE, cH_hist, cE_hist, idxBlk, Aidx, g5);
        % --- 简单收敛判定 ---
        delta_P = norm(p_cur - p_k_minus_1) / (norm(p_k_minus_1) + 1e-8);
        delta_C = norm([cH - cH_k_minus_1, cE - cE_k_minus_1], 'fro') / ...
                  (norm([cH_k_minus_1, cE_k_minus_1], 'fro') + 1e-8);

        fprintf('[REF BCD %02d/%02d] delta_P=%.3e, delta_C=%.3e\n', ...
                it, maxOuter, delta_P, delta_C);

        if it >= minOuter && delta_P < tol_relP && delta_C < tol_relC
            fprintf('[REF] Converged on reference image.\n');
            break;
        end

        p_k_minus_1  = p_cur;
        cH_k_minus_1 = cH;
        cE_k_minus_1 = cE;
    end

    % —— 得到最终参考光谱 —— 
    p_opt_ref = p_cur;
    [gH_ref, gE_ref, gR_ref, gG_ref, gB_ref, gMat_ref, den_ref] = get_gs_from_p(p_opt_ref, g5);

end
function [p_cur, cH, cE, cH_hist, cE_hist] = normalize_HE_RGB_state( ...
            p_cur, cH, cE, cH_hist, cE_hist, idxBlk, Aidx, g5)
    [gH_tmp, gE_tmp, gR_tmp, gG_tmp, gB_tmp, ~, ~] = get_gs_from_p(p_cur, g5);

    bH = idxBlk(1);
    sH = max(gH_tmp);
    if sH > 1e-9
        p_cur(bH(Aidx)) = p_cur(bH(Aidx)) / sH;
        cH = cH * sH;
        cH_hist = cH_hist * sH;
    end

    bE = idxBlk(2);
    sE = max(gE_tmp);
    if sE > 1e-9
        p_cur(bE(Aidx)) = p_cur(bE(Aidx)) / sE;
        cE = cE * sE;
        cE_hist = cE_hist * sE;
    end

    rgb_profiles = {gR_tmp(:), gG_tmp(:), gB_tmp(:)};
    for cc = 1:3
        gC = rgb_profiles{cc};
        kappaC = max(gC);
        if kappaC > 1e-9
            bC = idxBlk(cc + 2);   % blocks 3,4,5 -> R,G,B
            p_cur(bC(Aidx)) = p_cur(bC(Aidx)) / kappaC;
        end
    end
end
function [cH, cE, cH_hist, cE_hist] = solve_C_smooth_history( ...
            P, gH, gE, Gmat, den, ...
            cH_init, cE_init, ...
            cH_hist, cE_hist, ...
            optsPixJac, alpha_hist, alpha_update)
% C-step with robust OD scaling and history smoothing only.

    Ns = size(P, 2);
    cH = cH_init;
    cE = cE_init;
    den = max(den, 1e-12);

    nProbe = min(300, Ns);
    probeIdx = round(linspace(1, Ns, nProbe));
    res_probe = zeros(3, nProbe);

    for ii = 1:nProbe
        j = probeIdx(ii);
        x_probe = [cH_init(j); cE_init(j)];
        S_probe = gH*x_probe(1) + gE*x_probe(2);
        Y_probe = 10.^(-S_probe);
        T_probe = max(Gmat.' * Y_probe, 1e-12);
        F_probe = -log10(T_probe ./ den);
        res_probe(:, ii) = F_probe - P(:, j);
    end

    res_vec = res_probe(:);
    med_r = median(res_vec);
    sigma_od = 1.4826 * median(abs(res_vec - med_r));
    sigma_od = min(max(sigma_od, 0.01), 0.20);

    parfor j = 1:Ns
        Pj = P(:, j);
        x0 = [cH_init(j); cE_init(j)];
        c_hist_j = [cH_hist(j); cE_hist(j)];

        fun = @(x) pix_residual_history_with_jac( ...
            x, Pj, gH, gE, Gmat, den, alpha_hist, c_hist_j, sigma_od);

        x = lsqnonlin(fun, x0, [0;0], [5;5], optsPixJac);
        cH(j) = x(1);
        cE(j) = x(2);
    end

    cH_hist = (1 - alpha_update)*cH_hist + alpha_update*cH;
    cE_hist = (1 - alpha_update)*cE_hist + alpha_update*cE;
end

function [r, J] = pix_residual_history_with_jac(x, Pj, ...
                                                gH, gE, Gmat, den, ...
                                                alpha_hist, c_hist, sigma_od)
    den = max(den, 1e-12);

    S = gH*x(1) + gE*x(2);
    Y = 10.^(-S);
    T = max(Gmat.' * Y, 1e-12);
    F = -log10(T ./ den);

    r_data = (F - Pj) ./ max(sigma_od, 1e-2);

    SH = Gmat.' * (gH .* Y);
    SE = Gmat.' * (gE .* Y);
    J_data = [SH ./ T, SE ./ T] ./ max(sigma_od, 1e-2);

    s_hist = sqrt(alpha_hist);
    r_hist = s_hist * (x - c_hist);
    J_hist = s_hist * eye(2);

    r = [r_data; r_hist];
    J = [J_data; J_hist];
end
function [p_RGB_opt, best_score] = calibrate_camera_to_vectors(vH_tgt, vE_tgt, gH_ref, gE_ref, pRGB0, g5, lb, ub)
    % 简单的 fmincon 优化
    % 目标：调整 RGB 参数，使得 gH_ref 和 gE_ref 经过相机模型后的方向 与 vH_tgt, vE_tgt 一致
    
    % 归一化目标
    vH_tgt = vH_tgt / norm(vH_tgt);
    vE_tgt = vE_tgt / norm(vE_tgt);
    
    % Cost Function
    fun = @(p) cost_angle_mismatch(p, vH_tgt, vE_tgt, gH_ref, gE_ref, g5);
    
    opts = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp', ...
        'MaxIterations', 50, 'OptimalityTolerance', 1e-4);
    
    try
        [p_RGB_opt, best_score] = fmincon(fun, pRGB0, [],[],[],[], lb, ub, [], opts);
    catch
        p_RGB_opt = pRGB0; best_score = inf;
    end
end

function f = cost_angle_mismatch(pRGB, vH_t, vE_t, gH, gE, g5)
    pR = pRGB(1:15); pG = pRGB(16:30); pB = pRGB(31:45);
    sR = g5(pR); sG = g5(pG); sB = g5(pB);
    
    Gmat = [sR, sG, sB];
    den = sum(Gmat, 1)'; % 3x1
    
    % 模拟 H 向量 (假设浓度=1)
    transH = 10.^(-1.0 * gH);
    sigH = Gmat' * transH;
    odH = -log10(sigH ./ den);
    if norm(odH)>0, odH = odH/norm(odH); end
    
    % 模拟 E 向量
    transE = 10.^(-1.0 * gE);
    sigE = Gmat' * transE;
    odE = -log10(sigE ./ den);
    if norm(odE)>0, odE = odE/norm(odE); end
    
    % 损失：Cosine 距离
    f = (1 - dot(vH_t, odH)) + (1 - dot(vE_t, odE));
end
function [vH, vE] = get_HE_vectors_via_PCA(I_rgb)
    if isa(I_rgb, 'uint8'), I_rgb = double(I_rgb)/255; end
    I_rgb = max(I_rgb, 1e-4);
    OD = -log10(I_rgb);
    
    P = reshape(OD, [], 3);
    od_norm = sqrt(sum(P.^2, 2));
    mask = (od_norm > 0.15) & (od_norm < 2.5); % 简单阈值
    P_valid = P(mask, :);
    
    if size(P_valid, 1) < 100, vH=[1;0;0]; vE=[0;1;0]; return; end % Fallback
    
    [~, ~, V] = svd(P_valid, 'econ');
    P_2d = P_valid * V(:, 1:2);
    theta = atan2(P_2d(:,2), P_2d(:,1));
    [idx, ~] = kmeans(theta, 2, 'Replicates', 3, 'Distance', 'sqEuclidean');
    
    vecs = zeros(3, 2);
    for k=1:2
        mask_k = (idx==k);
        pts = P_valid(mask_k, :);
        norms_k = sqrt(sum(pts.^2,2));
        % 取该方向上浓度较大的点求均值，更鲁棒
        strong_mask = norms_k > quantile(norms_k, 0.7);
        if sum(strong_mask)>0
             vec_avg = mean(pts(strong_mask,:), 1)';
        else
             vec_avg = mean(pts, 1)';
        end
        vecs(:, k) = vec_avg / norm(vec_avg);
    end
    
    % 区分 H (红多) 和 E (绿/蓝多)
    if vecs(1,1) > vecs(1,2)
        vH = vecs(:,1); vE = vecs(:,2);
    else
        vH = vecs(:,2); vE = vecs(:,1);
    end
end
function [cH, cE, cH_hist, cE_hist, scaleInfo] = normalize_C_pixelwise_p95( ...
            cH, cE, cH_hist, cE_hist)

    eps_val = 1e-12;

    fgMask = (cH + cE) > eps_val;

    scaleInfo = struct();
    scaleInfo.used = true;
    scaleInfo.p95H = NaN;
    scaleInfo.p95E = NaN;

    if nnz(fgMask) <= 10
        return;
    end

    p95_h = prctile(cH(fgMask), 95);
    p95_e = prctile(cE(fgMask), 95);

    p95_h = max(p95_h, eps_val);
    p95_e = max(p95_e, eps_val);

    k_pixel = cH ./ p95_h + cE ./ p95_e;

    valid_mask = fgMask & (k_pixel >= eps_val);

    cH(valid_mask) = cH(valid_mask) ./ k_pixel(valid_mask);
    cE(valid_mask) = cE(valid_mask) ./ k_pixel(valid_mask);

    cH_hist(valid_mask) = cH_hist(valid_mask) ./ k_pixel(valid_mask);
    cE_hist(valid_mask) = cE_hist(valid_mask) ./ k_pixel(valid_mask);

    scaleInfo.used = true;
    scaleInfo.p95H = p95_h;
    scaleInfo.p95E = p95_e;
    scaleInfo.numValid = nnz(valid_mask);
end
function S = local_cost_with_fast_C_refit(p_all, P, cH_base, cE_base, g5, ...
                                          optsRefit, ...
                                          lambda_ortho, lambda_s, lambda_temp, ...
                                          gH_ref, gE_ref)

    % ===== 1) 当前 p_all 生成光谱 =====
    [gH, gE, gR, gG, gB, Gmat, den] = get_gs_from_p(p_all, g5);

    % ===== 2) 用当前 p_all 快速 refit C =====
    optsLocal = optsRefit;
    optsLocal.cH_init = cH_base(:);
    optsLocal.cE_init = cE_base(:);

    % P 是 3 x Ns，fast GN 函数输入是 Ns x 3
    [cH_tmp, cE_tmp] = solve_concentration_fast_GN_patch_init( ...
        P.', gH, gE, Gmat, den, optsLocal);

    cH_tmp = cH_tmp(:).';
    cE_tmp = cE_tmp(:).';

    % ===== 3) 用 refit 后的 C 计算数据项，并做尺度归一化 =====
    expo  = -(gH*cH_tmp + gE*cE_tmp);              % 300 x Ns
    ODhat = -log10( (Gmat.' * 10.^(expo)) ./ den ); % 3 x Ns
    
    eps_ = 1e-6;
    
    % 通道权重
    w = [1; 1; 1];
    
    diff   = (ODhat - P).^2;      % 3 x Ns
    diff_w = w .* diff;           % 3 x Ns
    
    num   = sqrt(sum(diff_w, 1));                 % 1 x Ns
    denom = max(eps_, sqrt(sum(P.^2, 1)));         % 1 x Ns
    
    relErr = num ./ denom;                         % 1 x Ns
    
    Res  = mean(relErr);
    core = Res / (1 + Res);                        % 压缩到 0~1
    
    S = core;
    
    
    % ===== 4) 可选：加入 H/E 光谱正则 =====
    if lambda_ortho > 0 || lambda_s > 0
        [cos2_val, smoothH, smoothE] = reg_terms(gH, gE);
    
        % cos2_val 本身通常在 0~1，不需要额外压缩
        R_ortho = lambda_ortho * cos2_val;
    
        % smoothH/smoothE 可能尺度较大，建议也压缩一下
        smoothRaw = smoothH + smoothE;
        smoothNorm = smoothRaw / (1 + smoothRaw);
    
        R_smooth = lambda_s * smoothNorm;
    
        S = S + R_ortho + R_smooth;
    end
    
    
    % ===== 5) 可选：加入模板正则，并做尺度归一化 =====
    if nargin >= 10 && ~isempty(gH_ref) && ~isempty(gE_ref) && lambda_temp > 0
    
        if numel(gH_ref) ~= numel(gH) || numel(gE_ref) ~= numel(gE)
            error('Template spectra size mismatch with gH/gE.');
        end
    
        diffH = gH(:) - gH_ref(:);
        diffE = gE(:) - gE_ref(:);
    
        tempRaw = norm(diffH, 2)^2 + norm(diffE, 2)^2;
    
        % 与旧版本保持一致：压缩到 0~1
        tempNorm = tempRaw / (1 + tempRaw);
    
        R_temp = lambda_temp * tempNorm;
    
        S = S + R_temp;
    end
end
function [c, ceq] = nonlcon_physical_HE_RGB_relation(p_all)

    [c_he, ~] = nonlcon_physical_HE_relation(p_all);

    idxBlk = @(k) (15*(k-1)+1):(15*k);
    g5 = @(p) Gaussian_reconstruction5(p);

    pR = p_all(idxBlk(3));
    pG = p_all(idxBlk(4));
    pB = p_all(idxBlk(5));

    gR = g5(pR); gR = gR(:);
    gG = g5(pG); gG = gG(:);
    gB = g5(pB); gB = gB(:);

    L = numel(gR);
    lambda_axis = linspace(400, 700, L)';

    centR = sum(lambda_axis .* gR.^2) / (sum(gR.^2) + 1e-12);
    centG = sum(lambda_axis .* gG.^2) / (sum(gG.^2) + 1e-12);
    centB = sum(lambda_axis .* gB.^2) / (sum(gB.^2) + 1e-12);

    delta_RGB = 10;  % minimum centroid separation [nm]

    % fmincon uses c <= 0: B + delta <= G, G + delta <= R.
    c_rgb_order = [
        centB + delta_RGB - centG;
        centG + delta_RGB - centR
    ];

    c = [c_he; c_rgb_order];
    ceq = [];
end
