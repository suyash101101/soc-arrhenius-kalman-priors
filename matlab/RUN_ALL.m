%% RUN_ALL.m -- batch runner for SOC_Pipeline_v3.m (v3.3)
%  Put this file next to SOC_Pipeline_v3.m and the data folders, choose a stage, run.
%  Each run writes results_<dataset>_<tag>/ and a console log console_<dataset>_<tag>.txt.
%  A failed run is logged and the batch continues. Finished runs (results_macros.tex present) are skipped.
%
%  v3.3 defaults for every run: random initial-SOC error of 10-20 % on every segment, SPKF alpha = 1,
%  SHEKF is the residual base, 5 seeds, whole-segment bootstrap, Coulomb-counting baselines, convergence time.
%
%  stage 'A' : MAIN PAPER, SHEKF base, extras ON: lg loco, samsung loco, samsung cross, lg loto, lg tempx, samsung tempx
%              (robustness only on the two loco runs; ECM-parameter bootstrap on the two loco runs)
%  stage 'B' : leave-one-standard-cycle-out (LG, US06 held out), SHEKF base, extras ON
%  stage 'C' : SPKF base (for the validation-block choice of the prior and the supplement tables), extras OFF,
%              the six main protocols and the cycle-out run
%  stage 'D' : nearest-temperature parameter lookup (SHEKF base, extras OFF, 1 seed): lg loto, lg tempx, samsung loto, samsung tempx
%  stage 'E' : SUPPLEMENT: lg sparse, samsung sparse, samsung loto (SHEKF + SPKF base), braun regime, braun wltpseen
%              (+ nearest lookup for the sparse runs)
%  Recommended order: A, C, B, D, E.  Stage A is the minimum that the main paper needs.
%  Edit the 'express' preset below if you want a quick smoke test first (3 seeds, shorter training).
stage  = 'SX';       % still-open batch
preset = 'full';       % 'full' | 'express'

% each row: dataset, protocol, baseFilter, paramMode, extras, nSeeds, doRobust, ecmBoot, holdCycle
R = {};
switch stage
  case 'A'
    R = {'lg','loco','shekf','arrhenius',true,5,true,20,'US06'; 'samsung','loco','shekf','arrhenius',true,5,true,20,'US06'; ...
         'samsung','cross','shekf','arrhenius',false,5,false,0,'US06'; ...
         'lg','loto','shekf','arrhenius',true,5,false,0,'US06'; 'lg','tempx','shekf','arrhenius',true,5,false,0,'US06'; ...
         'samsung','tempx','shekf','arrhenius',true,5,false,0,'US06'};
  case 'B'
    R = {'lg','lcyc','shekf','arrhenius',true,5,false,0,'US06'};
  case 'C'
    R = {'lg','loco','spkf','arrhenius',false,5,false,0,'US06'; 'samsung','loco','spkf','arrhenius',false,5,false,0,'US06'; ...
         'lg','loto','spkf','arrhenius',false,5,false,0,'US06'; 'lg','tempx','spkf','arrhenius',false,5,false,0,'US06'; ...
         'samsung','tempx','spkf','arrhenius',false,5,false,0,'US06'; 'lg','lcyc','spkf','arrhenius',false,5,false,0,'US06'};
  case 'D'
    R = {'lg','loto','shekf','nearest',false,1,false,0,'US06'; 'lg','tempx','shekf','nearest',false,1,false,0,'US06'; ...
         'samsung','loto','shekf','nearest',false,1,false,0,'US06'; 'samsung','tempx','shekf','nearest',false,1,false,0,'US06'};
  case 'SX'
    R = {'samsung','cross','shekf','arrhenius',true,5,false,0,'US06'};
  case 'N5'
    R = {'lg','loto','shekf','nearest',false,5,false,0,'US06'; ...
         'lg','tempx','shekf','nearest',false,5,false,0,'US06'; ...
         'samsung','loto','shekf','nearest',false,5,false,0,'US06'; ...
         'samsung','tempx','shekf','nearest',false,5,false,0,'US06'; ...
         'lg','sparse','shekf','nearest',false,5,false,0,'US06'; ...
         'samsung','sparse','shekf','nearest',false,5,false,0,'US06'};
  case 'OPEN'
    R = {'lg','loco','spkf','arrhenius',true,5,true,0,'US06'; ...
         'samsung','loco','spkf','arrhenius',true,5,true,0,'US06'; ...
         'samsung','tempx','spkf','arrhenius',true,5,false,0,'US06'; ...
         'lg','sparse','shekf','arrhenius',true,5,false,0,'US06'; ...
         'samsung','sparse','shekf','arrhenius',true,5,false,0,'US06'; ...
         'lg','sparse','spkf','arrhenius',true,5,false,0,'US06'; ...
         'samsung','sparse','spkf','arrhenius',true,5,false,0,'US06'; ...
         'lg','loto','spkf','arrhenius',true,5,false,0,'US06'; ...
         'lg','tempx','spkf','arrhenius',true,5,false,0,'US06'; ...
         'lg','lcyc','spkf','arrhenius',true,5,false,0,'US06'};
  case 'E'
    R = {'braun','regime','shekf','arrhenius',true,5,true,0,'US06'; 'braun','wltpseen','shekf','arrhenius',true,5,true,0,'US06'; ...
         'lg','sparse','shekf','arrhenius',false,5,false,0,'US06'; 'samsung','sparse','shekf','arrhenius',false,5,false,0,'US06'; ...
         'samsung','loto','shekf','arrhenius',false,5,false,0,'US06'; ...
         'lg','sparse','spkf','arrhenius',false,5,false,0,'US06'; 'samsung','sparse','spkf','arrhenius',false,5,false,0,'US06'; ...
         'samsung','loto','spkf','arrhenius',false,5,false,0,'US06'; ...
         'braun','regime','spkf','arrhenius',false,5,false,0,'US06'; 'braun','wltpseen','spkf','arrhenius',false,5,false,0,'US06'; ...
         'lg','sparse','shekf','nearest',false,1,false,0,'US06'; 'samsung','sparse','shekf','nearest',false,1,false,0,'US06'};
end
save('batch_state.mat', 'stage', 'preset', 'R');       % SOC_Pipeline_v3 clears the workspace, so the loop state lives in this file
nRuns = size(R,1);
for r = 1:nRuns
    S = load('batch_state.mat');  stage = S.stage;  preset = S.preset;  R = S.R;  nRuns = size(R,1);
    tag = R{r,2};
    if strcmp(R{r,2}, 'lcyc'), tag = ['lcyc_' lower(R{r,9})]; end
    if strcmp(R{r,4},'nearest'), tag = [tag '_nearest']; end
    if ~strcmp(R{r,3},'spkf'), tag = [tag '_base' R{r,3}]; end
    outd = sprintf('results_%s_%s', R{r,1}, tag);
    fprintf('\n##### [%d/%d] %s / %s / base %s / %s / extras %d / seeds %d #####\n', r, nRuns, R{r,1}, R{r,2}, R{r,3}, R{r,4}, R{r,5}, R{r,6});
    if exist(fullfile(outd,'results_macros.tex'),'file')
        fprintf('  already finished (%s), skipping\n', outd);  continue;
    end
    RUN = struct('dataset',R{r,1},'protocol',R{r,2},'baseFilter',R{r,3},'paramMode',R{r,4},'extras',R{r,5}, ...
                 'nSeeds',R{r,6},'doRobust',R{r,7},'ecmBoot',R{r,8},'holdCycle',R{r,9},'preset',preset); %#ok<NASGU>
    diary(sprintf('console_%s_%s.txt', R{r,1}, tag));
    try
        SOC_Pipeline_v3;                       % reads RUN, then clears everything else
    catch ME
        fprintf('RUN FAILED: %s\n', ME.message);  disp(getReport(ME, 'extended', 'hyperlinks', 'off'));
    end
    diary off;
    clear RUN;
end
S = load('batch_state.mat');  fprintf('\nBatch stage %s finished. Upload every results_* folder and console_*.txt.\n', S.stage);
