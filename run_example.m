function run_example(refImagePath, inputDir, outRoot)
%RUN_EXAMPLE Run the released H&E spectral/concentration solver.

if nargin ~= 3
    error('Usage: run_example(refImagePath, inputDir, outRoot)');
end
if ~isfile(refImagePath)
    error('Reference image not found: %s', refImagePath);
end
if ~isfolder(inputDir)
    error('Input directory not found: %s', inputDir);
end

codeDir = fileparts(mfilename('fullpath'));
oldDir = pwd;
cleanup = onCleanup(@() cd(oldDir));
cd(codeDir);
addpath(codeDir, '-begin');

he_spectral_decomposition( ...
    refImagePath, inputDir, outRoot, 100, 2, 0, ...
    'NumSample', 600, ...
    'ThrWhite', 240, ...
    'LowCut', 15, ...
    'lambda_orthoC', 0, ...
    'lambda_s', 0, ...
    'PatchVis', false);
clear cleanup
end
