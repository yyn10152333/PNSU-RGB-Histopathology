function od = rgb2od(rgb)
    epsilon = 1e-6;  % 避免 log(0)
    od = -log10((rgb + epsilon) / (255+ epsilon));
end