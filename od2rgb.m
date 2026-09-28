function rgb = od2rgb(od)
 epsilon = 1e-6;  % 避免 log(0)
    rgb = (255+epsilon) * 10.^(-od)-epsilon;
end