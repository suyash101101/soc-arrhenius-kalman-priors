%% ========================================================================
%  SOC_Pipeline_v3.m (v3.3)
%  Physics-informed hybrid SOC estimation on three public datasets:
%      Kalman priors (EKF / SPKF / IEKF / SHEKF)
%    + learners that correct a Kalman prior (cfg.baseFilter, default SPKF) (LSBoost, PI-LSTM, BiLSTM)
%    + simplex-constrained NNLS stacking over all seven estimators.
%
%  One script, three datasets, several evaluation protocols. Set the two
%  switches below, run, upload the results folder, change the switches, run again.
%
%  cfg.dataset   cfg.protocol   what it tests
%  -----------   ------------   ----------------------------------------------
%  'lg'          'loco'         Leave-cycle-out: train on Mixed1-6, early-stop and fit
%  'samsung'                    the stacking weights on Mixed7-8, test on UDDS, HWFET,
%                               LA92, US06; all six temperatures (+40 to -20 C), one
%                               global model, temperature is an input.
%  'lg'          'loto'         Leave one temperature out (default 0 C): the cell is never seen
%  'samsung'                    at that temperature; filters borrow the ECM of the nearest
%                               temperature, learners see temperature only as an input.
%  'lg'          'sparse'       Like loto with 25, 0 and -10 C held out: the cell is calibrated at
%  'samsung'                    only 40, 10 and -20 C (half of the calibration tests).
%  'lg'          'tempx'        Hard extrapolation: train on 10/25/40 C, test on 0/-10/-20 C.
%  'samsung'                    Expected to fail for every method; reported as a limitation.
%  'samsung'     'cross'        Cross-cell: LG-trained models (from results_lg_loco)
%                               applied to the Samsung cell with no refit.
%  'braun'       'regime'       Fresh cell: train on constant-current cycling, test on the
%                               last 65 % of the WLTP block (hard regime shift).
%  'braun'       'wltpseen'     Fresh cell: train on cycling + first 30 % of WLTP, validate
%                               on the next 20 %, test on the last 50 % of WLTP.
%
%  Data folders (unzip next to this script):
%      LG_HG2/        6 folders 0degC 10degC 25degC 40degC n10degC n20degC   (LG HG2, Kollmeyer)
%      Samsung_30T/   6 folders 0degC 10degC 25degC 40degC -10degC -20degC   (Samsung 30T)
%      Experimental_data_fresh_cell.csv, OCV_vs_SOC_curve.csv, Experimental_data_aged_cell.csv
%  Requirements: R2021b or newer, Deep Learning Toolbox, Statistics and Machine
%  Learning Toolbox. Trained networks are cached, so an interrupted run resumes.
%
%  Reference SOC
%   LG / Samsung : SOC = 1 + Ah/Cap, Ah = cycler's amp-hour counter (negative on
%                  discharge), Cap = |Ah| at the end of the file / 0.95, because the
%                  authors repeat each profile until 95 % of the 1C capacity at that
%                  temperature has been discharged. Every file starts fully charged.
%   Braun        : Coulomb counting at 19.96 Ah with the voltage-limit calibration of
%                  the dataset's own script.
%  Sign convention in this script: discharge current is POSITIVE.
%  ========================================================================
clearvars -except RUN; clc; close all; rng(42);   % RUN: optional struct set by RUN_ALL.m

cfg.dataset      = 'lg';       % 'lg' | 'samsung' | 'braun'
cfg.protocol     = 'loco';     % see table above
cfg.baseFilter   = 'shekf';     % Kalman prior the learners correct: 'spkf' | 'shekf' | 'iekf' | 'ekf'   (v3.3 default: SHEKF)
cfg.paramMode    = 'arrhenius'; % ECM parameters of temperatures without FIT data: 'arrhenius' (ln R vs 1/T) | 'nearest'
cfg.holdTemp     = 0;          % 'loto' only: the chamber temperature (C) that is held out
cfg.holdCycle    = 'US06';     % 'lcyc' only: the standard cycle (UDDS | HWFET | LA92 | US06) that is held out
cfg.tauMode      = 'arrhenius';% time constant of unmeasured temperatures: 'arrhenius' | 'nearest' | 'median'
% ---- v3.3: realistic start-up, SPKF scaling, statistics, evidence blocks ----
cfg.initErr      = 0.15;       % every segment (FIT, VAL, TEST) starts the filters and Coulomb counting from the true initial SOC
                               % minus/plus a pseudo-random error of 10-20 % (magnitude U(0.10,0.20), 15 % = mean); 0 = true initial SOC
cfg.initSeed     = 2024;
cfg.p0soc        = 1e-2;       % initial SOC variance of every filter (std 0.1); 1e-4 reproduces the v3.2 runs
cfg.ukfA         = 1;          % SPKF scaling alpha (v3.2 used 1e-3, which gives a zeroth covariance weight of about -1e6)
cfg.ukfK         = 1;          % SPKF kappa
cfg.diagSPKF     = true;       % legacy-vs-fixed SPKF comparison with min eigenvalue of P (filters only)
cfg.nSeeds       = 5;          % seeds for BiLSTM, PI-LSTM and the fusion (seed 1 = main run)
cfg.convTol      = 0.02;       % convergence band (SOC)
cfg.convHold     = 60;         % samples inside the band to count as converged
cfg.unitLen      = 1800;       % bootstrap unit (samples) for protocols with fewer than 8 test segments (Braun)
cfg.doRobust     = true;       % robustness scenarios (RUN_ALL: only the loco runs)
cfg.oracle       = true;       % parameter oracle / parameter-law ablation (filters only) for loto, tempx and sparse
cfg.ecmBoot      = 0;          % segment-bootstrap replicates of the identified ECM parameters (0 = off; RUN_ALL sets 20 for the loco runs)
cfg.cc.offsetA   = 0.1;        % realistic Coulomb counting: current offset = offsetA * capacity/3 A
cfg.cc.gain      = 0.01;       % ... and a 1 % gain error
cfg.extras       = true;       % lambda sweep, seeds, baselines, prior ablation, robustness, conformal
cfg.W            = 30;         % look-back window (samples = seconds at 1 Hz)
cfg.dt           = 1;          % s
cfg.trainStride  = 3;          % use every 3rd window for training (1 Hz windows overlap by 97 %)
cfg.valStride    = 5;          % use every 5th VAL window for early stopping
cfg.residual     = true;       % learners predict SOC - SPKF (correction of the Kalman prior)
cfg.resScale     = 10;         % target scaling for the neural nets
cfg.lambdaPhys   = 1.0;        % PI-LSTM physics weight when cfg.extras = false (else chosen on VAL)
cfg.lambdaGrid   = [0 0.1 1 10];
cfg.piEpochs     = 12;
cfg.nnEpochs     = 12;
cfg.nSeedsPI     = 3;          % kept for the express preset; superseded by cfg.nSeeds
cfg.bootB        = 1000;  cfg.bootBlock = 100;
cfg.nSingleRuns  = 200;
cfg.useCache     = true;
cfg.preset       = 'full';     % 'full' | 'express' (about 3x faster: stride 6, 8 epochs, lambda in {0,1}, 2 seeds)
cfg.dirBraun     = '.';
cfg.dirLG        = 'LG_HG2';
cfg.dirSamsung   = 'Samsung_30T';
cfg.CN_Ah_braun  = 19.96;
cfg.confLevel    = 0.90;       % split-conformal coverage target
cfg.robust.initErr   = -0.15;  % stress scenario: an additional 15 % SOC shift on top of the default initial error
cfg.robust.sigV      = 0.010;  % V noise (V)
cfg.robust.sigI      = 0.10;   % I noise (A) -- scaled by capacity/3 Ah for Braun
cfg.robust.biasI     = 0.10;   % current-sensor offset (A) -- scaled likewise

if exist('RUN','var') && isstruct(RUN)            % batch mode: RUN_ALL.m overrides the switches above
    fn = fieldnames(RUN);  for q = 1:numel(fn), cfg.(fn{q}) = RUN.(fn{q}); end
end

if strcmp(cfg.preset, 'express')
    cfg.trainStride = 6;  cfg.nnEpochs = 8;  cfg.piEpochs = 8;  cfg.lambdaGrid = [0 1];  cfg.nSeedsPI = 2;  cfg.nSeeds = 3;
end
if strcmp(cfg.protocol, 'sparse'), cfg.holdTemp = [25 0 -10]; end      % calibrate the ECM at 40, 10, -20 C only
runTag = cfg.protocol;  if strcmp(cfg.paramMode, 'nearest') && any(strcmp(cfg.protocol, {'loto','tempx','sparse'})), runTag = [runTag '_nearest']; end
baseIdx = find(strcmp(cfg.baseFilter, {'ekf','spkf','iekf','shekf'}));  assert(~isempty(baseIdx), 'cfg.baseFilter must be ekf|spkf|iekf|shekf');
baseTag = '';  if baseIdx ~= 2, baseTag = ['_base' cfg.baseFilter]; end
if strcmp(cfg.protocol, 'lcyc'), runTag = ['lcyc_' lower(cfg.holdCycle)]; end
if ~strcmp(cfg.tauMode, 'arrhenius'), runTag = [runTag '_tau' cfg.tauMode]; end
runTag = [runTag baseTag];
outDir = sprintf('results_%s_%s', cfg.dataset, runTag);
if ~exist(outDir,'dir'), mkdir(outDir); end
W = cfg.W;  dt = cfg.dt;
resS = cfg.resScale * cfg.residual + 1 * ~cfg.residual;     % scale used by the nets

%% ========================================================================
%  STEP 0: LOAD DATA, REFERENCE SOC, SPLIT, PER-TEMPERATURE ECM
%  ========================================================================
fprintf('=== STEP 0: %s / %s ===\n', cfg.dataset, cfg.protocol);
if strcmp(cfg.dataset, 'braun')
    [D, SEG, PAR] = loadBraunData(cfg);
else
    if strcmp(cfg.dataset, 'lg'), rootDir = cfg.dirLG; else, rootDir = cfg.dirSamsung; end
    [D, SEG, PAR] = loadKollmeyerData(rootDir, cfg);
end
N = numel(D.V);
M = cell(1, numel(PAR));
mo = struct('p0soc', cfg.p0soc, 'ukfA', cfg.ukfA, 'ukfK', cfg.ukfK);
for j = 1:numel(PAR), M{j} = makeMdl(PAR(j), dt, mo); end
segStart = [SEG.i0]';
% ---- initial SOC given to the estimators: true initial SOC + a deliberate error (v3.3 default) ----
nSegAll = numel(SEG);  rng(cfg.initSeed);  uE = rand(nSegAll, 2);  rng(42);
for a = 1:nSegAll
    e0 = (cfg.initErr > 0) * (0.10 + 0.10*uE(a,1)) / 0.15 * cfg.initErr * (2*(uE(a,2) > 0.5) - 1);
    s0 = SEG(a).soc0 + e0;
    if s0 > 1 || s0 < 0, s0 = SEG(a).soc0 - e0; end          % fully charged starts can only be wrong on the low side
    SEG(a).s0 = min(max(s0, 0), 1);
end
fprintf('  initial SOC error given to the estimators: mean |e0| %.3f, range [%.3f, %.3f]\n', ...
    mean(abs([SEG.s0] - [SEG.soc0])), min([SEG.s0] - [SEG.soc0]), max([SEG.s0] - [SEG.soc0]));
fprintf('  %d samples, %d segments | FIT %d | VAL %d | TEST %d samples\n', N, numel(SEG), ...
    sum(D.grp==1), sum(D.grp==2), sum(D.grp==3));
for j = 1:numel(PAR)
    fprintf('  ECM set %d (%.0f C): R0 = %.4f Ohm, R1 = %.4f Ohm, tau = %.1f s, cap = %.3f Ah, voltage residual RMS %.4f V\n', ...
        j, PAR(j).tempC, PAR(j).R0, PAR(j).R1, PAR(j).tau, PAR(j).cap, sqrt(PAR(j).Rn));
end

eF = find(D.grp==1 & D.pos>=W);  eV = find(D.grp==2 & D.pos>=W);  eT = find(D.grp==3 & D.pos>=W);
if strcmp(cfg.protocol, 'cross'), eF = zeros(0,1); eV = zeros(0,1); end   % cross-cell: nothing is trained here
eFtr = eF(1:cfg.trainStride:end);  eVs = eV(1:cfg.valStride:end);
nFit = numel(eF); nVal = numel(eV); nTest = numel(eT);
fprintf('  windows: FIT %d (train subsample %d) | VAL %d (early-stop subsample %d) | TEST %d\n', ...
    nFit, numel(eFtr), nVal, numel(eVs), nTest);

%% ========================================================================
%  STEP 3: KALMAN FILTERS (physics priors), PER SEGMENT, WITH TIMING
%  ========================================================================
fprintf('\n=== STEP 3: Kalman filters ===\n');
segUse = unique(D.seg(D.grp > 0))';
if strcmp(cfg.protocol, 'cross'), segUse = unique(D.seg(D.grp == 3))'; end
[PR, tKFms, DG] = runPriors(D.V, D.I, SEG, segUse, M, 0);
SOC_EKF = PR(:,1); SOC_UKF = PR(:,2); SOC_IEKF = PR(:,3); SOC_SHEKF = PR(:,4);
SOC_BASE = PR(:, baseIdx);          % Kalman prior that the learners correct (cfg.baseFilter)
fprintf('  per-step ms: EKF %.4f | SPKF %.4f | IEKF %.4f | SHEKF %.4f\n', tKFms);
fprintf('  min eigenvalue of P (all used segments): EKF %.3g | SPKF %.3g | IEKF %.3g | SHEKF %.3g ; SPKF Cholesky failures %d\n', colfun(DG.lmin, @min), sum(DG.ncf(:,2)));
gname = {'FIT','VAL','TEST'};
for b = 1:3
    ii = find(D.grp == b);  if isempty(ii) || (strcmp(cfg.protocol,'cross') && b < 3), continue; end
    fprintf('  %-4s SOC range [%.3f, %.3f] | RMSE EKF %.4f SPKF %.4f IEKF %.4f SHEKF %.4f\n', gname{b}, ...
        min(D.soc(ii)), max(D.soc(ii)), rms(D.soc(ii)-PR(ii,1)), rms(D.soc(ii)-PR(ii,2)), ...
        rms(D.soc(ii)-PR(ii,3)), rms(D.soc(ii)-PR(ii,4)));
end

%% ========================================================================
%  STEP 4: FEATURES (train-only z-score) AND WINDOWS
%  ========================================================================
fprintf('\n=== STEP 4: Features and windows ===\n');
[X8, X11] = makeFeatures(D.V, D.I, D.T, PR, segStart, baseIdx);
iFitRows = find(D.grp == 1);
if strcmp(cfg.protocol, 'cross')
    L = load(fullfile(['results_lg_loco' baseTag], 'models.mat'));     % LG-trained models, no refit
    mu8 = L.mu8; sig8 = L.sig8; mu11 = L.mu11; sig11 = L.sig11;
else
    [~, mu8,  sig8 ] = zscoreTrain(X8,  iFitRows);
    [~, mu11, sig11] = zscoreTrain(X11, iFitRows);
end
X8n = (X8 - mu8) ./ sig8;   X11n = (X11 - mu11) ./ sig11;   X6n = X8n(:, 1:6);
muI = mu11(2); sigI = sig11(2);
% Kalman prior used for residual learning = cfg.baseFilter (default SPKF = column 9 of the 11-feature set)
bc = 7 + baseIdx;                   % column of the base prior in the 11-feature set (EKF 8, SPKF 9, IEKF 10, SHEKF 11)
rc = struct('col', bc * cfg.residual, 'mu', mu11(bc), 'sg', sig11(bc), 'scale', resS);
% capacity (coulombs) of the parameter set that serves each sample's segment
QcAll = zeros(N,1);
for j = 1:numel(PAR), QcAll(D.par == j) = PAR(j).cap * 3600; end

tgtOf  = @(e) (D.soc(e) - cfg.residual * SOC_BASE(e)) * resS;       % net target
backOf = @(e, p) min(max(cfg.residual * SOC_BASE(e) + double(p(:)) / resS, 0), 1);
tgtB   = @(e) (D.soc(e) - cfg.residual * SOC_BASE(e));              % tree target
backB  = @(e, p) min(max(cfg.residual * SOC_BASE(e) + double(p(:)), 0), 1);

X3_8_F  = seq3(X8n, eFtr, W);   X3_8_V  = seq3(X8n, eVs, W);   X3_8_T  = seq3(X8n, eT, W);
X3_11_F = seq3(X11n, eFtr, W);  X3_11_V = seq3(X11n, eVs, W);  X3_11_T = seq3(X11n, eT, W);
YF = D.soc(eFtr); YV = D.soc(eV); YT = D.soc(eT); YVs = D.soc(eVs);
QF = QcAll(eFtr);
sig = {cfg.dataset, cfg.protocol, baseIdx, N, nFit, nVal, nTest, W, cfg.trainStride, cfg.residual, cfg.resScale, ...
       cfg.initErr, cfg.p0soc, cfg.ukfA, cfg.ukfK, cfg.tauMode, cfg.holdCycle, cfg.paramMode};   % cache key: any change retrains
Time_T = (1:nTest)';

%% ========================================================================
%  STEPS 5-7: LEARNERS (BiLSTM, PI-LSTM, LSBoost) -- correct the SPKF prior
%  ========================================================================
isCross = strcmp(cfg.protocol, 'cross');
cname = @(tag) sprintf('cache_%s_%s_%s.mat', cfg.dataset, runTag, tag);
tInf = struct();

fprintf('\n=== STEP 5: BiLSTM ===\n');
if ~isCross
    net_bi = loadOrTrain(cname('bilstm'), sig, cfg.useCache, @() trainBiLSTM( ...
        toCell(X3_8_F), tgtOf(eFtr), toCell(X3_8_V), tgtOf(eVs), 8, cfg.nnEpochs));
else
    net_bi = L.net_bi;
end
tic; P_bi_T = backOf(eT, predict(net_bi, toCell(X3_8_T), 'MiniBatchSize', 2048));
tInf.BiLSTM_batch = toc / nTest * 1e3;
if ~isCross
    P_bi_V = backOf(eV, predict(net_bi, toCell(seq3(X8n, eV, W)), 'MiniBatchSize', 2048));
end

fprintf('\n=== STEP 6: PI-LSTM (physics-weight sweep chosen on VAL) ===\n');
lamGrid = cfg.lambdaGrid;  if ~cfg.extras, lamGrid = [0 cfg.lambdaPhys]; end
piNets = cell(1, numel(lamGrid));  piValR = nan(1, numel(lamGrid));
X3_11_Vfull = [];
if ~isCross
    for q = 1:numel(lamGrid)
        rng(7);
        piNets{q} = loadOrTrain(cname(sprintf('pi_lam%g', lamGrid(q))), [sig, {cfg.piEpochs}], cfg.useCache, ...
            @() trainPILSTM(X3_11_F, YF, QF, X3_11_V, YVs, W, dt, muI, sigI, lamGrid(q), cfg.piEpochs, 128, rc));
        piValR(q) = rms(YVs - predictPI(piNets{q}, X3_11_V, rc));
        fprintf('  lambda = %-5g  VAL RMSE %.4f %%\n', lamGrid(q), 100*piValR(q));
    end
    posIdx = find(lamGrid > 0);
    [~, jb] = min(piValR(posIdx));  qBest = posIdx(jb);  lamBest = lamGrid(qBest);
    q0 = find(lamGrid == 0, 1);
    net_pi = piNets{qBest};
    fprintf('  PI-LSTM uses lambda = %g (lowest VAL RMSE among lambda > 0)\n', lamBest);
else
    net_pi = L.net_pi;  lamBest = NaN;  q0 = [];  qBest = [];
end
tic; P_pi_T = predictPI(net_pi, X3_11_T, rc);  tInf.PILSTM_batch = toc / nTest * 1e3;
if ~isCross
    X3_11_Vfull = seq3(X11n, eV, W);
    P_pi_V = predictPI(net_pi, X3_11_Vfull, rc);
end

fprintf('\n=== STEP 7: LSBoost ===\n');
if ~isCross
    tr = templateTree('MaxNumSplits', 20, 'MinLeafSize', 5);
    mdl_boost = fitrensemble(X8n(eFtr,:), tgtB(eFtr), 'Method', 'LSBoost', ...
        'NumLearningCycles', 300, 'Learners', tr, 'LearnRate', 0.05);
    mdl_boost = compact(mdl_boost);
else
    mdl_boost = L.mdl_boost;
end
tic; P_bo_T = backB(eT, predict(mdl_boost, X8n(eT,:)));  tInf.Boost_batch = toc / nTest * 1e3;
if ~isCross, P_bo_V = backB(eV, predict(mdl_boost, X8n(eV,:))); end

%% ========================================================================
%  STEP 8: NNLS STACKING OVER SEVEN ESTIMATORS (weights fitted on VAL only)
%  ========================================================================
fprintf('\n=== STEP 8: NNLS stacking ===\n');
names7 = {'EKF','SPKF','IEKF','SHEKF','LSBoost','PI-LSTM','BiLSTM'};
Phi_T = double([SOC_EKF(eT), SOC_UKF(eT), SOC_IEKF(eT), SOC_SHEKF(eT), P_bo_T, P_pi_T, P_bi_T]);
if ~isCross
    Phi_V = double([SOC_EKF(eV), SOC_UKF(eV), SOC_IEKF(eV), SOC_SHEKF(eV), P_bo_V, P_pi_V, P_bi_V]);
    w = nnlsSimplex(Phi_V, YV);
else
    w = L.w;
end
F_T = clip01(Phi_T * w);
for i = 1:7, fprintf('    %-8s w = %.4f\n', names7{i}, w(i)); end
if ~isCross
    F_V = clip01(Phi_V * w);
    save('models_tmp.mat', 'net_bi','net_pi','mdl_boost','w','mu8','sig8','mu11','sig11','rc');
    movefile('models_tmp.mat', fullfile(outDir, 'models.mat'));
end

%% ========================================================================
%  STEP 9: METRICS, LATENCY, STATISTICAL EVIDENCE
%  ========================================================================
fprintf('\n=== STEP 9: Metrics ===\n');
names8 = [names7, {'NNLS Fusion'}];
Pred_T = [Phi_T, F_T];
M8 = zeros(8,4);
for m = 1:8, M8(m,:) = metrics(YT, Pred_T(:,m)); end

% ---- single-sample latency (ms), median of repeated calls ----
x1c = toCell(X3_8_T(:,:,1)); x1p = X3_11_T(:,:,[1 1]);
lat.BiLSTM = medianTime(@() predict(net_bi, x1c), cfg.nSingleRuns);
lat.PILSTM = medianTime(@() predictPI(net_pi, x1p, rc), cfg.nSingleRuns);
lat.Boost  = medianTime(@() predict(mdl_boost, X8n(eT(1),:)), cfg.nSingleRuns);
lat.EKF = tKFms(1); lat.SPKF = tKFms(2); lat.IEKF = tKFms(3); lat.SHEKF = tKFms(4);
latVec = [lat.EKF lat.SPKF lat.IEKF lat.SHEKF lat.Boost lat.PILSTM lat.BiLSTM];
latFused = sum(latVec);
latAll = [latVec, latFused];
fprintf('  single-sample latency (ms): fused total %.3f\n', latFused);

% ---- uncertainty (v3.3): whole-segment bootstrap and paired Wilcoxon tests on per-segment RMSE ----
%  SOC errors drift over tens of minutes, so resampling 100-sample blocks understates the interval. The unit is a whole
%  test segment (one cycle at one temperature); a protocol with fewer than 8 test segments (Braun) is cut into 30-min units.
SE = (YT - Pred_T).^2;
segKey = D.seg(eT);
if numel(unique(segKey)) < 8, segKey = segKey * 1e6 + floor((D.pos(eT) - 1) / cfg.unitLen); end
rng(11);
[ciR, boots, unitRMSE, unitN] = segBoot(SE, segKey, cfg.bootB);
rng(11);  [ciBlock, ~] = blockBoot(SE, cfg.bootB, cfg.bootBlock);          % old moving-block interval, kept for comparison
dBoots = boots(:, 1:7) - boots(:, 8);
dCI = prctile(dBoots, [2.5 97.5], 1);
pW = ones(1,7);
for m = 1:7, pW(m) = signrank(unitRMSE(:,m), unitRMSE(:,8)); end
fprintf('  bootstrap / Wilcoxon units: %d (segments or 30-min blocks), median %d samples\n', size(unitRMSE,1), round(median(unitN)));

% ---- per-temperature and per-cycle breakdown of the TEST set ----
tcT = D.tempC(eT);  cyT = D.cyc(eT);
tempList = unique(tcT(~isnan(tcT)));   tempList = sort(tempList(:), 'descend');
perTemp = nan(numel(tempList), 8);
for a = 1:numel(tempList)
    ii = tcT == tempList(a);
    for m = 1:8, perTemp(a,m) = rms(YT(ii) - Pred_T(ii,m)); end
end
cycNames = {'UDDS','HWFET','LA92','US06','Mixed'};
perCyc = nan(numel(cycNames), 8);
for a = 1:numel(cycNames)
    ii = cyT == a;  if ~any(ii), continue; end
    for m = 1:8, perCyc(a,m) = rms(YT(ii) - Pred_T(ii,m)); end
end

% ---- contribution analysis for the fusion stage (refit on VAL) ----
abl = {'Full fusion (7 estimators)', 1:7; 'Without PI-LSTM', [1:5 7]; 'Without BiLSTM', 1:6; ...
       'Without LSBoost', [1:4 6 7]; 'Learners only (no Kalman filters)', 5:7; 'Kalman filters only', 1:4};
ablRes = zeros(size(abl,1), 3);
if ~isCross
    for a = 1:size(abl,1)
        cols = abl{a,2};  wa = nnlsSimplex(Phi_V(:,cols), YV);
        mm = metrics(YT, clip01(Phi_T(:,cols) * wa));  ablRes(a,:) = mm(1:3);
    end
end

% ---- why NNLS: alternative fusion rules (all fitted on VAL) ----
rules = {'NNLS, simplex (proposed)','Equal weights','Inverse-MSE weights', ...
         'Unconstrained least squares','Best single estimator on VAL', ...
         'NNLS, cross-fitted on VAL folds','NNLS, load-gated (two regimes)', ...
         'NNLS, temperature-gated'};
nRules = numel(rules);  ruleRes = nan(nRules, 4);  ruleW = cell(nRules, 1);
if ~isCross
    ruleW{1} = w;  ruleW{2} = ones(7,1)/7;
    iv = 1 ./ max(mean((YV - Phi_V).^2, 1)', eps);  ruleW{3} = iv / sum(iv);
    ruleW{4} = Phi_V \ YV;
    [~, ib] = min(mean((YV - Phi_V).^2, 1)); wb = zeros(7,1); wb(ib) = 1; ruleW{5} = wb;
    K = 5; fe = round(linspace(0, nVal, K+1)); Wk = zeros(7, K); cvSE = [];
    for k = 1:K
        ho = (fe(k)+1):fe(k+1);  tri = setdiff(1:nVal, ho);
        Wk(:,k) = nnlsSimplex(Phi_V(tri,:), YV(tri));
        cvSE = [cvSE; (YV(ho) - Phi_V(ho,:) * Wk(:,k)).^2]; %#ok<AGROW>
    end
    ruleW{6} = mean(Wk, 2);  cvRMSE_NNLS = sqrt(mean(cvSE));
    fprintf('  VAL cross-fit RMSE of NNLS: %.4f %% SOC\n', 100*cvRMSE_NNLS);
    for r = 1:6
        mm = metrics(YT, clip01(Phi_T * ruleW{r}));  ruleRes(r,:) = [mm(1) mm(2) min(ruleW{r}) sum(ruleW{r})];
    end
    % load-gated: two weight vectors split at the VAL median |current|
    thrI = median(abs(D.I(eV)));  gV = abs(D.I(eV)) > thrI;  gT = abs(D.I(eT)) > thrI;
    wLo = nnlsSimplex(Phi_V(~gV,:), YV(~gV));  wHi = nnlsSimplex(Phi_V(gV,:), YV(gV));
    yg = zeros(nTest,1);  yg(~gT) = Phi_T(~gT,:) * wLo;  yg(gT) = Phi_T(gT,:) * wHi;
    mm = metrics(YT, clip01(yg));  ruleRes(7,:) = [mm(1) mm(2) min([wLo; wHi]) mean([sum(wLo) sum(wHi)])];
    ruleW{7} = (wLo + wHi) / 2;
    % temperature-gated: one weight vector per VAL chamber temperature, a TEST sample
    % uses the vector of the nearest VAL temperature (by its measured case temperature)
    tcV = D.tempC(eV);  tgl = unique(tcV(~isnan(tcV)))';
    if numel(tgl) > 1
        Wg = zeros(7, numel(tgl));  tMed = zeros(1, numel(tgl));
        for a = 1:numel(tgl)
            ii = tcV == tgl(a);  Wg(:,a) = nnlsSimplex(Phi_V(ii,:), YV(ii));  tMed(a) = median(D.T(eV(ii)));
        end
        [~, near] = min(abs(D.T(eT) - tMed), [], 2);
        yt = sum(Phi_T .* Wg(:, near)', 2);
        mm = metrics(YT, clip01(yt));  ruleRes(8,:) = [mm(1) mm(2) min(Wg(:)) mean(sum(Wg,1))];
        ruleW{8} = mean(Wg, 2);
    else
        ruleRes(8,:) = ruleRes(1,:);  ruleW{8} = w;
    end
end

% ---- split-conformal prediction intervals on the fused SOC ----
confRows = {};
if ~isCross
    resV = abs(YV - F_V);  nV = numel(resV);
    qv = quantile(resV, min(1, ceil((nV+1) * cfg.confLevel) / nV));
    covAll = mean(abs(YT - F_T) <= qv);
    confRows(end+1,:) = {'All test data', 100*covAll, 100*2*qv};
    tcV = D.tempC(eV);  tgl = unique(tcV(~isnan(tcV)))';
    if numel(tgl) > 1
        tMed = arrayfun(@(g) median(D.T(eV(tcV == g))), tgl);
        qg = arrayfun(@(g) quantile(resV(tcV == g), min(1, ceil((sum(tcV==g)+1) * cfg.confLevel) / sum(tcV==g))), tgl);
        [~, near] = min(abs(D.T(eT) - tMed), [], 2);
        covT = abs(YT - F_T) <= qg(near)';
        confRows(end+1,:) = {'Temperature-conditional', 100*mean(covT), 100*2*mean(qg(near))};
    end
    for a = 1:numel(tempList)
        ii = tcT == tempList(a);
        confRows(end+1,:) = {sprintf('%g C (global interval)', tempList(a)), 100*mean(abs(YT(ii)-F_T(ii)) <= qv), 100*2*qv}; %#ok<AGROW>
    end
end

%% ========================================================================
%  STEP 9a: COMPONENT ABLATIONS, SEEDS, BASELINES, ROBUSTNESS  (cfg.extras)
%  ========================================================================
abl2 = {};  seedRows = {};  baseNames = {'LSTM','GRU','CNN-LSTM','LSTM+KF'};
baseM = [];  basePerTemp = [];  baseLat = [];  robustRows = {};  robNames = {};
if ~isCross
    q0 = find(lamGrid == 0, 1);
    if ~isempty(q0)
        pp0 = predictPI(piNets{q0}, X3_11_T, rc);
        abl2(end+1,:) = {'PI-LSTM without physics residual ($\lambda=0$)', rms(YT-pp0), M8(6,1)};
    end
end
if cfg.extras && ~isCross
    fprintf('\n--- Ablation: BiLSTM without Kalman priors (6 measured inputs, absolute target) ---\n');
    net_bi0 = loadOrTrain(cname('bilstm_noprior'), sig, cfg.useCache, @() trainBiLSTM( ...
        toCell(X3_8_F(1:6,:,:)), D.soc(eFtr), toCell(X3_8_V(1:6,:,:)), YVs, 6, cfg.nnEpochs));
    p0 = clip01(double(predict(net_bi0, toCell(X3_8_T(1:6,:,:)), 'MiniBatchSize', 2048)));
    abl2(end+1,:) = {'BiLSTM without Kalman priors', rms(YT-p0), M8(7,1)};

    fprintf('--- Ablation: BiLSTM with priors as inputs only (absolute target) ---\n');
    net_bi1 = loadOrTrain(cname('bilstm_abs'), sig, cfg.useCache, @() trainBiLSTM( ...
        toCell(X3_8_F), D.soc(eFtr), toCell(X3_8_V), YVs, 8, cfg.nnEpochs));
    p1 = clip01(double(predict(net_bi1, toCell(X3_8_T), 'MiniBatchSize', 2048)));
    abl2(end+1,:) = {'BiLSTM, priors as inputs only (no residual target)', rms(YT-p1), M8(7,1)};

    fprintf('--- Ablation: LSBoost without Kalman priors / without residual target ---\n');
    mb0 = fitrensemble(X6n(eFtr,:), D.soc(eFtr), 'Method','LSBoost','NumLearningCycles',300, ...
          'Learners', tr, 'LearnRate', 0.05);
    abl2(end+1,:) = {'LSBoost without Kalman priors', rms(YT - clip01(predict(mb0, X6n(eT,1:6)))), M8(5,1)};
    mb1 = fitrensemble(X8n(eFtr,:), D.soc(eFtr), 'Method','LSBoost','NumLearningCycles',300, ...
          'Learners', tr, 'LearnRate', 0.05);
    abl2(end+1,:) = {'LSBoost, priors as inputs only (no residual target)', rms(YT - clip01(predict(mb1, X8n(eT,:)))), M8(5,1)};

    % seed study (5 seeds) is in STEP 9c below

    fprintf('\n--- Reference baselines on the measured inputs only (no Kalman priors) ---\n');
    baseP = zeros(nTest, 4);  baseM = zeros(4, 4);  baseLat = zeros(1, 4);
    for b = 1:3
        fprintf('  %s\n', baseNames{b});
        nb = loadOrTrain(cname(['base_' baseNames{b}]), sig, cfg.useCache, @() trainSeqBaseline( ...
            baseNames{b}, toCell(X3_8_F(1:6,:,:)), D.soc(eFtr), toCell(X3_8_V(1:6,:,:)), YVs, 6, cfg.nnEpochs));
        baseP(:,b) = clip01(double(predict(nb, toCell(X3_8_T(1:6,:,:)), 'MiniBatchSize', 2048)));
        baseM(b,:) = metrics(YT, baseP(:,b));
        xb = toCell(X3_8_T(1:6,:,1));
        baseLat(b) = medianTime(@() predict(nb, xb), 50);
        if b == 1, nbLSTM = nb; end
    end
    % Literature-style hybrid (virtual-sensor fusion, cf. Previtali et al.): the raw-input LSTM output is a
    % pseudo-measurement fused with Coulomb counting in a scalar Kalman filter. Measurement variance from VAL,
    % process variance fixed; filter restarts at every segment. No test information is used.
    pValL = clip01(double(predict(nbLSTM, toCell(X3_8_V(1:6,:,:)), 'MiniBatchSize', 2048)));
    Rm = max(var(YVs - pValL), 1e-6);
    baseP(:,4) = netKF(baseP(:,1), D.I(eT), QcAll(eT), D.seg(eT), Rm, 1e-9, dt);
    baseM(4,:) = metrics(YT, baseP(:,4));
    baseLat(4) = baseLat(1);
    fprintf('  LSTM+KF (measurement variance %.2e from VAL)\n', Rm);
    basePerTemp = nan(numel(tempList), 3);
    for a = 1:numel(tempList)
        ii = tcT == tempList(a);
        for b = 1:4, basePerTemp(a,b) = rms(YT(ii) - baseP(ii,b)); end
    end

    if cfg.doRobust
        fprintf('\n--- Robustness: perturb the TEST segments, keep every trained model fixed ---\n');
        ctx = struct('baseIdx',baseIdx,'D',D,'SEG',SEG,'M',{M},'net_bi',net_bi,'net_pi',net_pi,'mdl_boost',mdl_boost,'w',w, ...
                     'mu8',mu8,'sig8',sig8,'mu11',mu11,'sig11',sig11,'rc',rc,'cfg',cfg,'eT',eT,'resS',resS);
        scI = median([PAR.cap]) / 3;
        scen = {'Nominal (default start-up error)', 0, 0, 0, 0, 0; ...
                sprintf('Extra initial SOC shift %+.0f points', 100*cfg.robust.initErr), cfg.robust.initErr, 0, 0, 0, 0; ...
                sprintf('Noise %g mV and %.2g A', 1e3*cfg.robust.sigV, cfg.robust.sigI*scI), 0, cfg.robust.sigV, cfg.robust.sigI*scI, 0, 0; ...
                sprintf('Current offset %+.2g A', cfg.robust.biasI*scI), 0, 0, 0, cfg.robust.biasI*scI, 0; ...
                'Current gain error +2 percent', 0, 0, 0, 0, 0.02};
        robRMSE = zeros(size(scen,1), 8);  robNames = scen(:,1);
        for s = 1:size(scen,1)
            rng(100 + s);
            Vp = D.V + scen{s,3} * randn(N,1);
            Ip = D.I * (1 + scen{s,6}) + scen{s,4} * randn(N,1) + scen{s,5};
            o = evalScenario(ctx, Vp, Ip, D.T, scen{s,2});
            robRMSE(s,:) = o.rmse;
            fprintf('  %-34s fused %.3f %% | SPKF %.3f %% | PI-LSTM %.3f %% | BiLSTM %.3f %%\n', scen{s,1}, ...
                100*o.rmse(8), 100*o.rmse(2), 100*o.rmse(6), 100*o.rmse(7));
        end
        fprintf('  sanity: nominal fused RMSE %.4f %% vs main run %.4f %%\n', 100*robRMSE(1,8), 100*M8(8,1));
    end
end
% Cross-cell: residual learners are transferred from LG and are not refit, but raw-input baselines for
% Table VI can still be trained on the Samsung Mixed FIT/VAL split (same TEST windows as the transfer).
if cfg.extras && isCross
    fprintf('\n--- Reference baselines on Samsung Mixed FIT/VAL (cross TEST; raw inputs only) ---\n');
    eFb = find(D.grp==1 & D.pos>=W);  eVb = find(D.grp==2 & D.pos>=W);
    if isempty(eFb) || isempty(eVb)
        fprintf('  SKIP baselines: no Mixed FIT/VAL windows on this split\n');
    else
        % Local z-score on Samsung Mixed FIT (do not reuse LG transfer normalisation for raw baselines)
        [~, mu6b, sig6b] = zscoreTrain(X8(:,1:6), find(D.grp==1));
        X6b = (X8(:,1:6) - mu6b) ./ sig6b;
        eFtrB = eFb(1:cfg.trainStride:end);  eVsB = eVb(1:cfg.valStride:end);
        YVsB = D.soc(eVsB);
        X3b_F = seq3(X6b, eFtrB, W);  X3b_V = seq3(X6b, eVsB, W);  X3b_T = seq3(X6b, eT, W);
        baseP = zeros(nTest, 4);  baseM = zeros(4, 4);  baseLat = zeros(1, 4);
        for b = 1:3
            fprintf('  %s\n', baseNames{b});
            nb = loadOrTrain(cname(['base_' baseNames{b}]), sig, cfg.useCache, @() trainSeqBaseline( ...
                baseNames{b}, toCell(X3b_F), D.soc(eFtrB), toCell(X3b_V), YVsB, 6, cfg.nnEpochs));
            baseP(:,b) = clip01(double(predict(nb, toCell(X3b_T), 'MiniBatchSize', 2048)));
            baseM(b,:) = metrics(YT, baseP(:,b));
            xb = toCell(X3b_T(:,:,1));
            baseLat(b) = medianTime(@() predict(nb, xb), 50);
            if b == 1, nbLSTM = nb; end
        end
        pValL = clip01(double(predict(nbLSTM, toCell(X3b_V), 'MiniBatchSize', 2048)));
        Rm = max(var(YVsB - pValL), 1e-6);
        baseP(:,4) = netKF(baseP(:,1), D.I(eT), QcAll(eT), D.seg(eT), Rm, 1e-9, dt);
        baseM(4,:) = metrics(YT, baseP(:,4));
        baseLat(4) = baseLat(1);
        fprintf('  LSTM+KF (measurement variance %.2e from VAL)\n', Rm);
        basePerTemp = nan(numel(tempList), 4);
        for a = 1:numel(tempList)
            ii = tcT == tempList(a);
            for b = 1:4, basePerTemp(a,b) = rms(YT(ii) - baseP(ii,b)); end
        end
        for b = 1:numel(baseNames), fprintf('  %-10s RMSE %.3f %%\n', baseNames{b}, 100*baseM(b,1)); end
    end
end

%% ========================================================================
%  STEP 9b: FRESH-TO-AGED TRANSFER (Braun only, nothing refitted)
%  ========================================================================
MA = [];
if strcmp(cfg.dataset, 'braun') && isfile('Experimental_data_aged_cell.csv') && ~isCross
    fprintf('\n=== STEP 9b: aged-cell transfer ===\n');
    da = readtable('Experimental_data_aged_cell.csv');
    Na = numel(da.Time);
    soc_a = refSOC(da.Time, da.Current, da.Voltage, cfg.CN_Ah_braun);
    Va = smoothdata(da.Voltage, 'movmean', 5);  Ia = smoothdata(da.Current, 'movmean', 5);
    iWa = wltpStart(soc_a);
    Da = struct('V',Va,'I',Ia,'T',da.Temperature,'soc',soc_a,'seg',ones(Na,1),'pos',(1:Na)', ...
                'par',ones(Na,1),'tempC',nan(Na,1),'cyc',zeros(Na,1),'grp',zeros(Na,1));
    SEGa = struct('i0',1,'i1',Na,'par',1,'soc0',soc_a(1),'s0',max(soc_a(1) - 0.15*(cfg.initErr > 0), 0),'name','aged','tempC',NaN,'cyc','WLTP','grp',0,'cap',cfg.CN_Ah_braun);
    ctxA = struct('baseIdx',baseIdx,'D',Da,'SEG',SEGa,'M',{M},'net_bi',net_bi,'net_pi',net_pi,'mdl_boost',mdl_boost,'w',w, ...
                  'mu8',mu8,'sig8',sig8,'mu11',mu11,'sig11',sig11,'rc',rc,'cfg',cfg, ...
                  'eT',((iWa+W):Na)','resS',resS);
    oA = evalScenario(ctxA, Va, Ia, da.Temperature, 0);
    MA = [oA.rmse', oA.mae', oA.max'];
    fprintf('  aged WLTP block starts at %.2f h, %d windows\n', da.Time(iWa)/3600, numel(ctxA.eT));
    for m = 1:8, fprintf('  %-12s RMSE %.3f %% | MAE %.3f %%\n', names8{m}, 100*MA(m,1), 100*MA(m,2)); end
    fid = fopen(fullfile(outDir,'results_aged_rows.tex'),'w');
    for m = 1:8
        if m == 5 || m == 8, fprintf(fid, '\\midrule\n'); end
        fprintf(fid, '%s & %.3f & %.3f & %.3f & %.3f \\\\\n', texName(names8{m}), ...
                100*M8(m,1), 100*MA(m,1), 100*MA(m,2), 100*MA(m,3));
    end
    fprintf(fid, '\\bottomrule\n'); fclose(fid);
end

%% ========================================================================
%  STEP 10: CONSOLE REPORT
%  ========================================================================
fprintf('\n%-14s | RMSE %% | MAE %% | MAX %% |    R2   | RMSE 95%% CI        | ms\n', 'Method');
fprintf('%s\n', repmat('-', 1, 86));
for m = 1:8
    fprintf('%-14s | %6.3f | %5.3f | %5.3f | %7.3f | [%6.3f, %6.3f] | %.4f\n', names8{m}, ...
        100*M8(m,1), 100*M8(m,2), 100*M8(m,3), M8(m,4), 100*ciR(1,m), 100*ciR(2,m), latAll(m));
end
fprintf('\nPaired Wilcoxon p-values (|error|, estimator vs fusion):\n');
for m = 1:7, fprintf('  %-8s p = %.3g | dRMSE 95%% CI [%.3f, %.3f] %%\n', names7{m}, pW(m), 100*dCI(1,m), 100*dCI(2,m)); end
if ~isempty(tempList)
    fprintf('\nTEST RMSE (%%) by temperature:\n  %-6s', 'T/C'); fprintf(' %9s', names8{:}); fprintf('\n');
    for a = 1:numel(tempList), fprintf('  %-6g', tempList(a)); fprintf(' %9.3f', 100*perTemp(a,:)); fprintf('\n'); end
end
if ~isempty(baseM)
    fprintf('\nReference baselines (measured inputs only): ');
    for b = 1:numel(baseNames), fprintf('%s %.3f %% | ', baseNames{b}, 100*baseM(b,1)); end, fprintf('\n');
end
if ~isempty(confRows)
    fprintf('\nSplit-conformal intervals (target %.0f %% coverage):\n', 100*cfg.confLevel);
    for a = 1:size(confRows,1), fprintf('  %-34s coverage %.1f %% | width %.2f %% SOC\n', confRows{a,1}, confRows{a,2}, confRows{a,3}); end
end

%% ========================================================================
%  STEP 9c (v3.3): SEEDS, COULOMB COUNTING, CONVERGENCE, PARAMETER ORACLE,
%                  ECM PARAMETER SPREAD, SPKF DIAGNOSTIC, VALIDATION RMSE, TUNING LOG
%  ========================================================================
fprintf('\n=== STEP 9c: v3.3 evidence blocks ===\n');
segAll = unique(D.seg(D.grp > 0))';
if isCross, segAll = unique(D.seg(D.grp == 3))'; end

% ---------- (1) five seeds for BiLSTM, PI-LSTM, fusion (seed 1 = main run) ----------
nS = cfg.nSeeds;
NB = cell(1,nS);  NP = cell(1,nS);  NP0 = cell(1,nS);  WS = nan(7,nS);
NB{1} = net_bi;  NP{1} = net_pi;  WS(:,1) = w;
if ~isCross && ~isempty(q0), NP0{1} = piNets{q0}; end
if ~isCross
    for s = 2:nS
        fprintf('  seed %d of %d: BiLSTM, PI-LSTM (lambda %g)%s\n', s, nS, lamBest, repmat(' and PI-LSTM (lambda 0)', 1, double(cfg.extras)));
        NB{s} = loadOrTrain(cname(sprintf('bilstm_s%d', s)), [sig, {cfg.nnEpochs, s}], cfg.useCache, @() trainBiLSTMs(1000*s+1, ...
                toCell(X3_8_F), tgtOf(eFtr), toCell(X3_8_V), tgtOf(eVs), 8, cfg.nnEpochs));
        NP{s} = loadOrTrain(cname(sprintf('pi_s%d_k1', s)), [sig, {cfg.piEpochs, lamBest, s}], cfg.useCache, @() trainPILSTMs(1000*s+2, ...
                X3_11_F, YF, QF, X3_11_V, YVs, W, dt, muI, sigI, lamBest, cfg.piEpochs, 128, rc));
        if cfg.extras
            NP0{s} = loadOrTrain(cname(sprintf('pi_s%d_k2', s)), [sig, {cfg.piEpochs, 0, s}], cfg.useCache, @() trainPILSTMs(1000*s+3, ...
                    X3_11_F, YF, QF, X3_11_V, YVs, W, dt, muI, sigI, 0, cfg.piEpochs, 128, rc));
        end
    end
elseif exist('L','var') && isfield(L, 'bi_seeds')
    for s = 2:min(nS, numel(L.bi_seeds)), NB{s} = L.bi_seeds{s};  NP{s} = L.pi_seeds{s};  WS(:,s) = L.w_seeds(:,s); end
end
X3_8_Vfull = [];  if ~isCross, X3_8_Vfull = seq3(X8n, eV, W); end
seedR = nan(nS, 5);       % columns: LSBoost, PI-LSTM, BiLSTM, fusion, PI-LSTM without physics residual
for s = find(~cellfun(@isempty, NB))
    Pb_T = backOf(eT, predict(NB{s}, toCell(X3_8_T), 'MiniBatchSize', 2048));
    Pp_T = predictPI(NP{s}, X3_11_T, rc);
    PhiTs = double([Phi_T(:,1:4), P_bo_T, Pp_T, Pb_T]);
    if ~isCross
        Pb_V = backOf(eV, predict(NB{s}, toCell(X3_8_Vfull), 'MiniBatchSize', 2048));
        Pp_V = predictPI(NP{s}, X3_11_Vfull, rc);
        WS(:,s) = nnlsSimplex(double([Phi_V(:,1:4), P_bo_V, Pp_V, Pb_V]), YV);
    end
    Fs = clip01(PhiTs * WS(:,s));
    seedR(s,1:4) = [rms(YT - P_bo_T), rms(YT - Pp_T), rms(YT - Pb_T), rms(YT - Fs)];
    if ~isempty(NP0{s}), seedR(s,5) = rms(YT - predictPI(NP0{s}, X3_11_T, rc)); end
end
okS = find(~isnan(seedR(:,3)));
seedMean = colfun(seedR(okS,:), @mean);  seedSD = colfun(seedR(okS,:), @(v) std(v, 0));
fprintf('  seeds used: %d | RMSE %% mean +- sd: LSBoost %.3f(det.) | PI-LSTM %.3f +- %.3f | BiLSTM %.3f +- %.3f | Fusion %.3f +- %.3f\n', ...
    numel(okS), 100*seedMean(1), 100*seedMean(2), 100*seedSD(2), 100*seedMean(3), 100*seedSD(3), 100*seedMean(4), 100*seedSD(4));
if ~isCross
    bi_seeds = NB;  pi_seeds = NP;  w_seeds = WS;
    save(fullfile(outDir, 'models.mat'), 'bi_seeds', 'pi_seeds', 'w_seeds', '-append');
    if any(~isnan(seedR(:,5)))
        seedRows = {sprintf('PI-LSTM with physics residual ($\\lambda=%g$)', lamBest), seedR(okS,2); ...
                    'PI-LSTM without physics residual ($\lambda=0$)', seedR(okS(~isnan(seedR(okS,5))),5)};
    end
end
% estimator table: mean and sd over seeds (filters and LSBoost are deterministic)
seedEst = zeros(8,2);
seedEst(1:4,1) = M8(1:4,1);  seedEst(5,1) = seedMean(1);  seedEst(6,1) = seedMean(2);  seedEst(7,1) = seedMean(3);  seedEst(8,1) = seedMean(4);
seedEst(5:8,2) = seedSD(1:4)';
fid = fopen(fullfile(outDir,'results_seedtab_rows.tex'), 'w');
for m = 1:8, fprintf(fid, '%s & %.3f & %.3f & %d \\\\\n', texName(names8{m}), 100*seedEst(m,1), 100*seedEst(m,2), numel(okS)); end
fprintf(fid, '\\bottomrule\n');  fclose(fid);
fid = fopen(fullfile(outDir,'results_seeds_all.csv'), 'w');
fprintf(fid, 'seed,LSBoost,PILSTM,BiLSTM,Fusion,PILSTM_lambda0\n');
for s = okS', fprintf(fid, '%d,%.6f,%.6f,%.6f,%.6f,%.6f\n', s, seedR(s,:)); end
fclose(fid);

% ---------- (2) pure Coulomb counting (the reference is itself Coulomb counted, so the ideal case is near-exact) ----------
scI = median([PAR.cap]) / 3;
QtrueAll = zeros(N,1);  for a = 1:numel(SEG), QtrueAll(SEG(a).i0:SEG(a).i1) = SEG(a).cap * 3600; end
s0true = [SEG.soc0]';  s0est = [SEG.s0]';
ccNames = {'Coulomb counting, ideal (true initial SOC and capacity)', ...
           'Coulomb counting, initial SOC error only', ...
           'Coulomb counting, realistic (initial error, offset, gain)'};
ccCfg = {s0true, QtrueAll, 0, 0; s0est, QcAll, 0, 0; s0est, QcAll, cfg.cc.gain, cfg.cc.offsetA * scI};
ccP = zeros(nTest, 3);  ccM = zeros(3, 4);
for v = 1:3
    sf = ccRun(D.I, ccCfg{v,2}, SEG, ccCfg{v,1}, segAll, ccCfg{v,3}, ccCfg{v,4}, dt, N);
    ccP(:,v) = sf(eT);  ccM(v,:) = metrics(YT, ccP(:,v));
    fprintf('  %-62s RMSE %.3f %% | MAE %.3f %% | max %.2f %%\n', ccNames{v}, 100*ccM(v,1), 100*ccM(v,2), 100*ccM(v,3));
end
fid = fopen(fullfile(outDir,'results_cc_rows.tex'), 'w');
for v = 1:3, fprintf(fid, '%s & %.3f & %.3f & %.2f \\\\\n', ccNames{v}, 100*ccM(v,1), 100*ccM(v,2), 100*ccM(v,3)); end
fprintf(fid, '\\bottomrule\n');  fclose(fid);
if exist('baseP', 'var') && ~isempty(baseP) && size(baseP,1) == nTest
    fprintf('  LSTM+KF capacity: QcAll(eT) = capacity of the parameter set that serves each test segment (identified, or Arrhenius-extrapolated), i.e. the capacity the filters use\n');
end

% ---------- (3) convergence from the wrong initial SOC ----------
APall = [Pred_T, ccP];  apNames = [names8, {'CC ideal', 'CC init. error', 'CC realistic'}];
if exist('baseP', 'var') && ~isempty(baseP) && size(baseP,1) == nTest, APall = [APall, baseP];  apNames = [apNames, baseNames]; end
[cvMed, cvFrac] = convStats(abs(repmat(YT, 1, size(APall,2)) - APall), D.seg(eT), D.pos(eT), cfg.convTol, cfg.convHold);
fid = fopen(fullfile(outDir,'results_conv_rows.tex'), 'w');
for m = 1:numel(apNames)
    if isnan(cvMed(m)), cs = '--'; else, cs = sprintf('%.0f', cvMed(m)); end
    fprintf(fid, '%s & %s & %.0f\\%% \\\\\n', texName(apNames{m}), cs, 100*cvFrac(m));
end
fprintf(fid, '\\bottomrule\n');  fclose(fid);
fprintf('  convergence into +-%.0f %% for %d s: median time (s) / %% of segments that converge\n', 100*cfg.convTol, cfg.convHold);
for m = 1:numel(apNames), fprintf('    %-18s %6.0f s | %3.0f %%\n', apNames{m}, cvMed(m), 100*cvFrac(m)); end

% ---------- (4) parameter oracle and parameter-law ablation (filters only) ----------
if cfg.oracle && any(strcmp(cfg.protocol, {'loto','tempx','sparse'})) && ~strcmp(cfg.paramMode, 'nearest') && ~strcmp(cfg.dataset, 'braun')
    fprintf('\n--- Parameter oracle: replace one extrapolated parameter at a time by its directly identified value (filters only) ---\n');
    OR = paramOracle(D, SEG, PAR, M, cfg, mo, rootDir);
    if OR.ok
        fid = fopen(fullfile(outDir,'results_oracle_rows.tex'), 'w');
        fidc = fopen(fullfile(outDir,'results_oracle.csv'), 'w');  fprintf(fidc, 'tempC,variant,EKF,SPKF,IEKF,SHEKF\n');
        for a = 1:numel(OR.temps)
            if ~OR.have(a), continue; end
            for v = 1:numel(OR.names)
                fprintf(fid,  '$%g$ & %s & %.2f & %.2f & %.2f & %.2f \\\\\n', OR.temps(a), OR.names{v}, 100*OR.R(v,:,a));
                fprintf(fidc, '%g,%s,%.4f,%.4f,%.4f,%.4f\n', OR.temps(a), strrep(OR.names{v}, ',', ';'), 100*OR.R(v,:,a));
            end
            fprintf(fid, '\\midrule\n');
            fprintf('  %g C: parameters ext. -> direct: R0 %.4f -> %.4f | R1 %.4f -> %.4f | tau %.1f -> %.1f | Q %.3f -> %.3f\n', OR.prm(a,1:9));
            for v = 1:numel(OR.names), fprintf('     %-28s SPKF %6.2f %% | SHEKF %6.2f %% | EKF %6.2f %%\n', OR.names{v}, 100*OR.R(v,2,a), 100*OR.R(v,4,a), 100*OR.R(v,1,a)); end
        end
        fprintf(fid, '\\bottomrule\n');  fclose(fid);  fclose(fidc);
    end
end

% ---------- (5) spread of the identified ECM parameters ----------
if ~strcmp(cfg.dataset, 'braun') || true
    fid = fopen(fullfile(outDir,'results_ecmspread_rows.tex'), 'w');
    fidt = fopen(fullfile(outDir,'results_ecm_starts.csv'), 'w');  fprintf(fidt, 'tempC,start,R0_mOhm,R1_mOhm,tau_s,cost\n');
    for j = 1:numel(PAR)
        if isempty(PAR(j).allS), continue; end
        aS = PAR(j).allS;  [cb, ib] = min(aS(:,4));  agree = aS(:,4) <= 1.01 * cb;
        for q = 1:size(aS,1), fprintf(fidt, '%g,%d,%.3f,%.3f,%.2f,%.6g\n', PAR(j).tempC, q, 1e3*aS(q,1), 1e3*aS(q,2), aS(q,3), aS(q,4)); end
        segj = find([SEG.grp] == 1 & [SEG.par] == j);  capR = [min([SEG(segj).cap]) max([SEG(segj).cap])];
        ciE = nan(2,3);
        if cfg.ecmBoot > 0 && ~isempty(segj) && ~isnan(PAR(j).tempC)
            ocvf = @(s) interp1(PAR(j).ocvG, PAR(j).ocvV, max(0,min(1,s)), 'linear', 'extrap');
            Vc = cell(1,numel(segj));  Ic = Vc;  Sc = Vc;
            for q = 1:numel(segj), ii = SEG(segj(q)).i0:SEG(segj(q)).i1;  Vc{q} = D.V(ii);  Ic{q} = D.I(ii);  Sc{q} = D.soc(ii); end
            ciE = ecmBootstrap(Vc, Ic, Sc, ocvf, dt, cfg.ecmBoot);
        end
        fprintf(fid, '$%g$ & %.1f (%.1f--%.1f) & %.1f (%.1f--%.1f) & %.1f (%.1f--%.1f) & %.3f--%.3f & %d/%d \\\\\n', PAR(j).tempC, ...
            1e3*PAR(j).R0, 1e3*min(aS(agree,1)), 1e3*max(aS(agree,1)), 1e3*PAR(j).R1, 1e3*min(aS(agree,2)), 1e3*max(aS(agree,2)), ...
            PAR(j).tau, min(aS(agree,3)), max(aS(agree,3)), capR, sum(agree), size(aS,1));
        fprintf('  %g C: R0 %.4f R1 %.4f tau %.1f | starts within 1%% of best cost: %d of %d (R1 range %.4f--%.4f) | bootstrap 95%% R0 [%.4f %.4f] R1 [%.4f %.4f] tau [%.1f %.1f]\n', ...
            PAR(j).tempC, PAR(j).R0, PAR(j).R1, PAR(j).tau, sum(agree), size(aS,1), min(aS(agree,2)), max(aS(agree,2)), ciE(1,1), ciE(2,1), ciE(1,2), ciE(2,2), ciE(1,3), ciE(2,3));
    end
    fprintf(fid, '\\bottomrule\n');  fclose(fid);  fclose(fidt);
    % every parameter set, identified or extrapolated
    fid = fopen(fullfile(outDir,'results_ecm_rows.tex'), 'w');
    [~, ord] = sort([PAR.tempC], 'ascend');
    for j = ord
        fprintf(fid, '$%g$ & %.1f & %.1f & %.1f & %.3f & %.1f & %s \\\\\n', PAR(j).tempC, 1e3*PAR(j).R0, 1e3*PAR(j).R1, PAR(j).tau, PAR(j).cap, ...
            1e3*sqrt(PAR(j).Rn), ternary(isempty(PAR(j).allS), 'Arrhenius', 'identified'));
    end
    fprintf(fid, '\\bottomrule\n');  fclose(fid);
end

% ---------- (6) SPKF diagnostic: legacy scaling (alpha = 1e-3, kappa = 0) versus the fixed scaling ----------
if cfg.diagSPKF && ~isCross
    fprintf('\n--- SPKF diagnostic: min eigenvalue of P, legacy versus fixed sigma-point scaling ---\n');
    DGr = zeros(numel(segAll), 13);
    for a = 1:numel(segAll)
        k = segAll(a);  ii = SEG(k).i0:SEG(k).i1;  m0 = M{SEG(k).par};
        ml = m0;  ml.ukfA = 1e-3;  ml.ukfK = 0;   mf = m0;  mf.ukfA = cfg.ukfA;  mf.ukfK = cfg.ukfK;
        [sl, dl] = runUKF(ml, D.V(ii), D.I(ii), SEG(k).s0);   [sf, df] = runUKF(mf, D.V(ii), D.I(ii), SEG(k).s0);
        el = D.soc(ii) - sl;  ef = D.soc(ii) - sf;
        DGr(a,:) = [k, SEG(k).grp, SEG(k).tempC, dl, rms(el), max(abs(el)), df, rms(ef), max(abs(ef))];
    end
    fid = fopen(fullfile(outDir,'results_spkf_diag.csv'), 'w');
    fprintf(fid, 'segment,group,tempC,leg_lmin,leg_nbad,leg_ncholfail,leg_rmse,leg_maxerr,fix_lmin,fix_nbad,fix_ncholfail,fix_rmse,fix_maxerr\n');
    fprintf(fid, '%d,%d,%g,%.6g,%d,%d,%.6f,%.6f,%.6g,%d,%d,%.6f,%.6f\n', DGr');  fclose(fid);
    legBad = DGr(:,5) > 0 | DGr(:,6) > 0;  fixBad = DGr(:,9+1) > 0 | DGr(:,9+2) > 0;
    bigL = DGr(:,8) > 0.20;  bigF = DGr(:,13) > 0.20;
    fid = fopen(fullfile(outDir,'results_spkf_diag_rows.tex'), 'w');
    fprintf(fid, 'Legacy ($\\alpha_u=10^{-3}$, $\\kappa=0$) & %d of %d & %.2g & %.2f & %.1f & %d \\\\\n', sum(legBad), numel(legBad), min(DGr(:,4)), 100*median(DGr(:,7)), 100*max(DGr(:,8)), sum(bigL));
    fprintf(fid, 'Fixed ($\\alpha_u=%g$, $\\kappa=%g$) & %d of %d & %.2g & %.2f & %.1f & %d \\\\\n', cfg.ukfA, cfg.ukfK, sum(fixBad), numel(fixBad), min(DGr(:,9)), 100*median(DGr(:,12)), 100*max(DGr(:,13)), sum(bigF));
    fprintf(fid, '\\bottomrule\n');  fclose(fid);
    fprintf('  legacy: %d of %d segments had a non-PD P or a failed Cholesky, min eig %.3g, %d segments with max error > 20 %% (%d of them with a non-PD event)\n', ...
        sum(legBad), numel(legBad), min(DGr(:,4)), sum(bigL), sum(bigL & legBad));
    fprintf('  fixed : %d of %d segments, min eig %.3g, %d segments with max error > 20 %%\n', sum(fixBad), numel(fixBad), min(DGr(:,9)), sum(bigF));
end
% minimum eigenvalue of P for the four filters in the main run (fixed scaling)
fid = fopen(fullfile(outDir,'results_pmin.txt'), 'w');
fprintf(fid, 'min eigenvalue of P over all used segments: EKF %.3g | SPKF %.3g | IEKF %.3g | SHEKF %.3g\n', colfun(DG.lmin, @min));
fprintf(fid, 'steps with lambda_min <= 0: EKF %d | SPKF %d | IEKF %d | SHEKF %d ; SPKF Cholesky failures %d\n', sum(DG.nbad, 1), sum(DG.ncf(:,2)));
fclose(fid);  type(fullfile(outDir,'results_pmin.txt'));

% ---------- (7) validation RMSE of every estimator (the prior is chosen on these numbers, never on TEST) ----------
if ~isCross
    fid = fopen(fullfile(outDir,'results_valrmse.csv'), 'w');
    fprintf(fid, 'estimator,val_rmse_pct\n');
    for m = 1:7, fprintf(fid, '%s,%.5f\n', strrep(names7{m}, ',', ';'), 100*rms(YV - Phi_V(:,m))); end
    fprintf(fid, 'NNLS Fusion (fitted on VAL),%.5f\n', 100*rms(YV - F_V));
    fprintf(fid, 'base_filter,%s\n', cfg.baseFilter);  fclose(fid);
end

% ---------- (8) tuning log ----------
fid = fopen(fullfile(outDir,'results_tuning.txt'), 'w');
fprintf(fid, 'process noise Qn (SOC, V_RC): diag(%g, %g)\n', PAR(1).Qn(1,1), PAR(1).Qn(2,2));
fprintf(fid, 'measurement noise Rn = identified voltage-residual variance per parameter set (V^2): %s\n', mat2str([PAR.Rn], 4));
fprintf(fid, 'initial covariance P0 = diag(%g, %g); initial SOC error given: +-U(0.10,0.20) (cfg.initErr = %g)\n', cfg.p0soc, 1e-4, cfg.initErr);
fprintf(fid, 'SPKF: alpha_u %g, beta 2, kappa %g\n', cfg.ukfA, cfg.ukfK);
fprintf(fid, 'IEKF: at most 5 iterations, tolerance 1e-6; SHEKF: forgetting factor 0.98, R bounded to [0.2, 20] x Rn, Q and bias fixed\n');
fprintf(fid, 'identified at %s C\n', mat2str([PAR.tempC]));
fclose(fid);

%% ========================================================================
%  STEP 11: LATEX SNIPPETS, MACROS AND AUTOMATIC CLAIM CHECK
%  ========================================================================
fprintf('\n=== STEP 11: Writing LaTeX snippets ===\n');
if ~isCross, EV = YV - Phi_V;  Ecorr = corrcoef(EV); else, Ecorr = nan(7); end
[~, ibest]   = min(M8(1:7,1));
[~, ibestKF] = min(M8(1:4,1));
[~, ibestL]  = min(M8(5:7,1)); ibestL = ibestL + 4;

% ---- main table ----
fid = fopen(fullfile(outDir,'results_main_rows.tex'),'w');
cm = {100*M8(:,1), 100*M8(:,2), 100*M8(:,3), M8(:,4), latAll(:)};
bestIdx = [argmin(M8(:,1)) argmin(M8(:,2)) argmin(M8(:,3)) argmax(M8(:,4)) argmin(latAll(:))];
for m = 1:8
    cell5 = cell(1,5);
    for c = 1:5
        s = sprintf('%.3f', cm{c}(m));
        if bestIdx(c) == m && c <= 4, s = ['\textbf{' s '}']; end
        cell5{c} = s;
    end
    ci = sprintf('[%.3f, %.3f]', 100*ciR(1,m), 100*ciR(2,m));
    if m == 5 || m == 8, fprintf(fid, '\\midrule\n'); end
    fprintf(fid, '%s & %s & %s & %s & %s & %s & %s \\\\\n', texName(names8{m}), cell5{1}, ci, cell5{2}, cell5{3}, cell5{4}, cell5{5});
end
fprintf(fid, '\\bottomrule\n'); fclose(fid);

% ---- weights ----
fid = fopen(fullfile(outDir,'results_weights_rows.tex'),'w');
for i = 1:7, fprintf(fid, '%s & %.4f & %.2f \\\\\n', texName(names7{i}), w(i), 100*w(i)); end
fprintf(fid, '\\midrule\nSum & %.4f & %.2f \\\\\n\\bottomrule\n', sum(w), 100*sum(w)); fclose(fid);

% ---- per temperature / per cycle / baselines ----
if ~isempty(tempList)
    fid = fopen(fullfile(outDir,'results_pertemp_rows.tex'),'w');
    for a = 1:numel(tempList)
        fprintf(fid, '$%g$', tempList(a));
        for m = [2 5 6 7 8], fprintf(fid, ' & %.2f', 100*perTemp(a,m)); end
        if ~isempty(basePerTemp), fprintf(fid, ' & %.2f', 100*basePerTemp(a,1:3)); else, fprintf(fid, ' & -- & -- & --'); end
        fprintf(fid, ' \\\\\n');
    end
    fprintf(fid, '\\bottomrule\n'); fclose(fid);
    fid = fopen(fullfile(outDir,'results_percycle_rows.tex'),'w');
    for a = 1:numel(cycNames)
        if all(isnan(perCyc(a,:))), continue; end
        fprintf(fid, '%s', cycNames{a});
        for m = [1 2 5 6 7 8], fprintf(fid, ' & %.2f', 100*perCyc(a,m)); end
        fprintf(fid, ' \\\\\n');
    end
    fprintf(fid, '\\bottomrule\n'); fclose(fid);
end
if ~isempty(baseM)
    fid = fopen(fullfile(outDir,'results_baselines_rows.tex'),'w');
    for b = 1:numel(baseNames)
        fprintf(fid, '%s & %.3f & %.3f & %.3f & %.3f \\\\\n', baseNames{b}, 100*baseM(b,1), 100*baseM(b,2), 100*baseM(b,3), baseLat(b));
    end
    fprintf(fid, '\\midrule\n');
    for m = [2 6 7 8]
        fprintf(fid, '%s (this work) & %.3f & %.3f & %.3f & %.3f \\\\\n', texName(names8{m}), 100*M8(m,1), 100*M8(m,2), 100*M8(m,3), latAll(m));
    end
    fprintf(fid, '\\bottomrule\n'); fclose(fid);
end

if ~isCross
    % ---- fusion-rule, fusion-ablation, component, seed, lambda, robustness, conformal ----
    fid = fopen(fullfile(outDir,'results_fusion_rows.tex'),'w');
    for r = 1:nRules
        fprintf(fid, '%s & %.3f & %.3f & %.3f & %.3f \\\\\n', rules{r}, 100*ruleRes(r,1), 100*ruleRes(r,2), ruleRes(r,3), ruleRes(r,4));
    end
    fprintf(fid, '\\bottomrule\n'); fclose(fid);

    fid = fopen(fullfile(outDir,'results_ablation_rows.tex'),'w');
    for a = 1:size(abl,1), fprintf(fid, '%s & %.3f & %.3f & %.3f \\\\\n', abl{a,1}, 100*ablRes(a,1), 100*ablRes(a,2), 100*ablRes(a,3)); end
    fprintf(fid, '\\bottomrule\n'); fclose(fid);

    fid = fopen(fullfile(outDir,'results_component_rows.tex'),'w');
    for a = 1:size(abl2,1)
        fprintf(fid, '%s & %.3f & %.3f & %.1f\\%% \\\\\n', abl2{a,1}, 100*abl2{a,2}, 100*abl2{a,3}, 100*(abl2{a,2}-abl2{a,3})/abl2{a,2});
    end
    fprintf(fid, '\\bottomrule\n'); fclose(fid);

    fid = fopen(fullfile(outDir,'results_seeds_rows.tex'),'w');
    for a = 1:size(seedRows,1)
        v = 100*seedRows{a,2};  fprintf(fid, '%s & %.3f & %.3f & %d \\\\\n', seedRows{a,1}, mean(v), std(v), numel(v));
    end
    fprintf(fid, '\\bottomrule\n'); fclose(fid);

    fid = fopen(fullfile(outDir,'results_lambda_rows.tex'),'w');
    for q = 1:numel(lamGrid)
        fprintf(fid, '%g & %.3f & %.3f \\\\\n', lamGrid(q), 100*piValR(q), 100*rms(YT - predictPI(piNets{q}, X3_11_T, rc)));
    end
    fprintf(fid, '\\bottomrule\n'); fclose(fid);

    if ~isempty(robNames)
        fid = fopen(fullfile(outDir,'results_robust_rows.tex'),'w');
        for s = 1:numel(robNames)
            fprintf(fid, '%s', strrep(robNames{s}, '%', '\%'));
            for m = [1 2 5 6 7 8], fprintf(fid, ' & %.2f', 100*robRMSE(s,m)); end
            fprintf(fid, ' \\\\\n');
        end
        fprintf(fid, '\\bottomrule\n'); fclose(fid);
    end
    if ~isempty(confRows)
        fid = fopen(fullfile(outDir,'results_conformal_rows.tex'),'w');
        for a = 1:size(confRows,1), fprintf(fid, '%s & %.1f & %.2f \\\\\n', confRows{a,1}, confRows{a,2}, confRows{a,3}); end
        fprintf(fid, '\\bottomrule\n'); fclose(fid);
    end
end

% ---- macros (prefix identifies the run, e.g. \resLgLocoFusedRMSE) ----
cap1 = @(s) [upper(s(1)) s(2:end)];
pfx = ['res' cap1(cfg.dataset) cap1(cfg.protocol) repmat('Nearest', 1, ~isempty(strfind(runTag, '_nearest'))) repmat(['Base' cap1(cfg.baseFilter)], 1, baseIdx ~= 2) repmat(cap1(lower(cfg.holdCycle)), 1, strcmp(cfg.protocol, 'lcyc'))];
mac = struct();
mac.NTrain = nFit;  mac.NVal = nVal;  mac.NTest = nTest;
mac.FusedRMSE = sprintf('%.3f',100*M8(8,1));  mac.FusedMAE = sprintf('%.3f',100*M8(8,2));
mac.FusedMax  = sprintf('%.3f',100*M8(8,3));  mac.FusedRtwo = sprintf('%.4f',M8(8,4));
mac.FusedCILo = sprintf('%.3f',100*ciR(1,8)); mac.FusedCIHi = sprintf('%.3f',100*ciR(2,8));
mac.BestSingleName = texName(names7{ibest});  mac.BestSingleRMSE = sprintf('%.3f',100*M8(ibest,1));
mac.GainBest = sprintf('%.1f',100*(M8(ibest,1)-M8(8,1))/M8(ibest,1));
mac.BestKFName = texName(names7{ibestKF});    mac.BestKFRMSE = sprintf('%.3f',100*M8(ibestKF,1));
mac.BestLearnerName = texName(names7{ibestL}); mac.BestLearnerRMSE = sprintf('%.3f',100*M8(ibestL,1));
mac.GainKF = sprintf('%.1f',100*(M8(ibestKF,1)-M8(8,1))/M8(ibestKF,1));
mac.GainLearner = sprintf('%.1f',100*(M8(ibestL,1)-M8(8,1))/M8(ibestL,1));
mac.EKFRMSE = sprintf('%.3f',100*M8(1,1));   mac.SPKFRMSE = sprintf('%.3f',100*M8(2,1));
mac.IEKFRMSE = sprintf('%.3f',100*M8(3,1));  mac.SHEKFRMSE = sprintf('%.3f',100*M8(4,1));
mac.BoostRMSE = sprintf('%.3f',100*M8(5,1)); mac.PILSTMRMSE = sprintf('%.3f',100*M8(6,1));
mac.BiLSTMRMSE = sprintf('%.3f',100*M8(7,1));
for i = 1:7, mac.(['Weight' strrep(names7{i},'-','')]) = sprintf('%.3f', w(i)); end
mac.LatFused = sprintf('%.2f',latFused);  mac.LatBiLSTM = sprintf('%.2f',lat.BiLSTM);
mac.LatPILSTM = sprintf('%.2f',lat.PILSTM); mac.LatBoost = sprintf('%.2f',lat.Boost);
mac.LatKF = sprintf('%.3f',max([lat.EKF lat.SPKF lat.IEKF lat.SHEKF]));
mac.DeltaLo = sprintf('%.3f',100*dCI(1,ibest)); mac.DeltaHi = sprintf('%.3f',100*dCI(2,ibest));
mac.PvalBest = pstr(pW(ibest));
if ~isCross
    mac.LambdaBest = sprintf('%g', lamBest);
    mac.RuleOLS = sprintf('%.3f',100*ruleRes(4,1)); mac.RuleEqual = sprintf('%.3f',100*ruleRes(2,1));
    mac.RuleInv = sprintf('%.3f',100*ruleRes(3,1)); mac.RuleCross = sprintf('%.3f',100*ruleRes(6,1));
    mac.RuleGate = sprintf('%.3f',100*ruleRes(7,1)); mac.RuleTGate = sprintf('%.3f',100*ruleRes(8,1));
    mac.OLSMinW = sprintf('%.3f',ruleRes(4,3));    mac.OLSSumW = sprintf('%.3f',ruleRes(4,4));
    mac.AblNoPILSTM = sprintf('%.3f',100*ablRes(2,1));
    mac.MeanAbsErrCorr = sprintf('%.2f', mean(abs(Ecorr(5:7,5:7)), 'all', 'omitnan'));
    for a = 1:size(abl2,1), mac.(['Ablx' char('A'+a-1)]) = sprintf('%.3f',100*abl2{a,2}); end
    if ~isempty(seedRows)
        mac.SeedMeanPhys = sprintf('%.3f',100*mean(seedRows{1,2})); mac.SeedStdPhys = sprintf('%.3f',100*std(seedRows{1,2}));
        mac.SeedMeanNoPhys = sprintf('%.3f',100*mean(seedRows{2,2})); mac.SeedStdNoPhys = sprintf('%.3f',100*std(seedRows{2,2}));
    end
    if ~isempty(baseM)
        for b = 1:numel(baseNames), mac.(['Base' strrep(strrep(baseNames{b},'-',''),'+','Plus')]) = sprintf('%.3f',100*baseM(b,1)); end
    end
    if ~isempty(confRows), mac.ConfCover = sprintf('%.1f',confRows{1,2}); mac.ConfWidth = sprintf('%.2f',confRows{1,3}); end
    if ~isempty(robNames)
        for s = 2:numel(robNames), mac.(['RobFused' char('A'+s-2)]) = sprintf('%.3f',100*robRMSE(s,8)); mac.(['RobSPKF' char('A'+s-2)]) = sprintf('%.3f',100*robRMSE(s,2)); end
    end
end
if ~isempty(MA)
    mac.AgedFusedRMSE = sprintf('%.3f',100*MA(8,1)); mac.AgedBiLSTMRMSE = sprintf('%.3f',100*MA(7,1));
    mac.AgedBoostRMSE = sprintf('%.3f',100*MA(5,1)); mac.AgedPILSTMRMSE = sprintf('%.3f',100*MA(6,1));
    mac.AgedEKFRMSE = sprintf('%.3f',100*MA(1,1));  mac.AgedSPKFRMSE = sprintf('%.3f',100*MA(2,1));
end
if ~isempty(tempList)
    for a = 1:numel(tempList)
        tn = strrep(sprintf('%g', tempList(a)), '-', 'm');  tn = strrep(tn, '.', 'p');
        tn(isstrprop(tn,'digit')) = char(tn(isstrprop(tn,'digit')) - '0' + 'a');   % LaTeX macro names cannot hold digits: 40 -> eka
        mac.(['T' tn 'Fused']) = sprintf('%.2f', 100*perTemp(a,8));
        mac.(['T' tn 'SPKF'])  = sprintf('%.2f', 100*perTemp(a,2));
    end
end
fid = fopen(fullfile(outDir,'results_macros.tex'),'w');
fn = fieldnames(mac);
for i = 1:numel(fn)
    v = mac.(fn{i});  if isnumeric(v), v = sprintf('%d', v); end
    fprintf(fid, '\\newcommand{\\%s%s}{%s}\n', pfx, fn{i}, v);
end
fclose(fid);

% ---- automatic claim check ----
fid = fopen(fullfile(outDir,'results_claims.txt'),'w');
cl = @(ok, s) fprintf(fid, '[%s] %s\n', ternary(ok,'HOLDS ','FAILS '), s);
cl(all(M8(8,1) <= M8(1:7,1)), 'Fusion has the lowest RMSE of all eight estimators');
cl(all(M8(8,2) <= M8(1:7,2)), 'Fusion has the lowest MAE of all eight estimators');
cl(dCI(1,ibest) > 0, sprintf('Fusion beats best single estimator (%s): bootstrap CI of dRMSE excludes 0', names7{ibest}));
cl(all(pW < 0.05), 'Paired Wilcoxon p < 0.05 against all seven estimators');
cl(w(6) > 0.01, 'PI-LSTM receives non-trivial NNLS weight (> 1%)');
cl(M8(8,1) < M8(ibestKF,1), sprintf('Fusion beats the best Kalman filter (%s)', names7{ibestKF}));
if ~isempty(baseM), cl(M8(8,1) < min(baseM(:,1)), 'Fusion beats the LSTM, GRU, CNN-LSTM and LSTM+KF baselines'); end
if ~isCross
    cl(ablRes(2,1) >= ablRes(1,1), 'Adding PI-LSTM to the fusion does not increase RMSE');
    cl(ruleRes(1,1) <= min(ruleRes(2:4,1)), 'Simplex NNLS <= equal weights, inverse-MSE and unconstrained LS');
    cl(ruleRes(6,1) <= ruleRes(1,1), 'Cross-fitted NNLS <= plain NNLS on TEST');
    cl(ruleRes(7,1) <= ruleRes(1,1), 'Load-gated NNLS <= plain NNLS on TEST');
    cl(ruleRes(8,1) <= ruleRes(1,1), 'Temperature-gated NNLS <= plain NNLS on TEST');
    cl(ruleRes(4,3) < 0 || abs(ruleRes(4,4)-1) > 1e-3, 'Unconstrained LS yields negative or non-unit-sum weights');
    if size(abl2,1) >= 3
        cl(abl2{2,2} > abl2{2,3}, 'Kalman priors reduce BiLSTM RMSE');
        cl(abl2{3,2} > abl2{3,3}, 'Residual target reduces BiLSTM RMSE');
        cl(abl2{4,2} > abl2{4,3}, 'Kalman priors reduce LSBoost RMSE');
    end
    cl(abl2{1,2} > abl2{1,3}, 'Physics residual reduces PI-LSTM RMSE (single seed)');
    if ~isempty(seedRows), cl(mean(seedRows{1,2}) < mean(seedRows{2,2}), 'Physics residual lowers PI-LSTM RMSE on average over seeds'); end
    if ~isempty(robNames), cl(all(robRMSE(2:end,8) <= robRMSE(2:end,2)), 'Fusion at least as robust as the SPKF in every perturbation scenario'); end
    if ~isempty(confRows), cl(abs(confRows{1,2} - 100*cfg.confLevel) < 3, 'Split-conformal coverage within 3 points of target on TEST'); end
end
cl(latFused < 100, 'Fused single-sample latency < 100 ms');
fclose(fid);
type(fullfile(outDir,'results_claims.txt'));

%% ========================================================================
%  STEP 13: SAVE RAW RESULTS
%  ========================================================================
T = array2table([D.seg(eT), D.pos(eT), D.tempC(eT), D.cyc(eT), YT(:), Pred_T], 'VariableNames', ...
    [{'Segment','Pos','TempC','CycleCode','SOC_true'}, matlab.lang.makeValidName(names8)]);
writetable(T, fullfile(outDir,'predictions_test.csv'));
save(fullfile(outDir,'results_workspace.mat'), 'unitRMSE','unitN','ciBlock','seedR','seedEst','ccM','cvMed','cvFrac','apNames','M8','ciR','dCI','pW','w','latAll','names8','names7','ablRes', ...
     'ruleRes','rules','abl2','perTemp','perCyc','tempList','cfg','PAR','confRows','robNames','baseM','seedRows','MA');

%% ========================================================================
%  STEP 12: IEEE FIGURES
%  ========================================================================
fprintf('\n=== STEP 12: Figures ===\n');
try, theme('light'); catch, end
try
    set(groot, 'defaultFigureColor','w', 'defaultAxesColor','w', 'defaultAxesXColor','k', ...
        'defaultAxesYColor','k', 'defaultAxesZColor','k', 'defaultTextColor','k', ...
        'defaultAxesGridColor',[0.15 0.15 0.15], 'defaultAxesGridAlpha',0.25, ...
        'defaultLegendTextColor','k', 'defaultLegendColor','w', 'defaultLegendEdgeColor','k');
    set(groot, 'defaultAxesFontName','Times New Roman', 'defaultTextFontName','Times New Roman', ...
        'defaultAxesFontSize',8, 'defaultLegendFontSize',7, 'defaultAxesLineWidth',0.6, ...
        'defaultLineLineWidth',1.0, 'defaultAxesTickDir','in', 'defaultAxesBox','on');
catch ME
    warning('Could not set all graphics defaults: %s', ME.message);
end
pal = [0.62 0.62 0.62; 0.48 0.48 0.48; 0.34 0.34 0.34; 0.20 0.20 0.20; ...
       0.00 0.45 0.70; 0.84 0.37 0.00; 0.00 0.62 0.45; 0.80 0.00 0.00];

try
% Fig 1: tracking on representative test segments (warmest UDDS and coldest segment)
segT = D.seg(eT);  segIds = unique(segT)';
if numel(segIds) == 1
    segShow = segIds;
else
    tcS = arrayfun(@(s) SEG(s).tempC, segIds);   isU = arrayfun(@(s) strcmp(SEG(s).cyc,'UDDS'), segIds);
    cand = segIds(isU);  if isempty(cand), cand = segIds; end
    tc2 = arrayfun(@(s) SEG(s).tempC, cand);
    [~, i1] = min(abs(tc2 - 25));  [~, i2] = min(tc2);
    segShow = unique([cand(i1), cand(i2)], 'stable');
end
nC = numel(segShow);
f = newFig(3.5 * min(nC,2) + 0.2, 5.8);
for c = 1:nC
    ii = find(segT == segShow(c));  stp = max(1, floor(numel(ii)/2500));  ii = ii(1:stp:end);
    tt = D.pos(eT(ii)) / 60;
    ttl = ternary(nC > 1, sprintf('%s, %g C', SEG(segShow(c)).cyc, SEG(segShow(c)).tempC), 'TEST block');
    subplot(3, nC, c); hold on; grid on;
    plot(tt, YT(ii), 'k-', 'LineWidth', 1.6);
    plot(tt, Pred_T(ii,6), '-', 'Color', pal(6,:), 'LineWidth', 0.7);
    plot(tt, Pred_T(ii,7), '-', 'Color', pal(7,:), 'LineWidth', 0.7);
    plot(tt, Pred_T(ii,8), '--', 'Color', pal(8,:), 'LineWidth', 1.1);
    ylabel('SOC'); title(['(a) ' ttl ': learners and fusion'], 'FontWeight','normal');
    if c == 1, legend({'Reference','PI-LSTM','BiLSTM','NNLS fusion'}, 'Location','best'); end
    subplot(3, nC, nC + c); hold on; grid on;
    plot(tt, YT(ii), 'k-', 'LineWidth', 1.6);
    lsKF = {'-','--',':','-.'};
    for m = 1:4, plot(tt, Pred_T(ii,m), 'Color', pal(m,:), 'LineStyle', lsKF{m}); end
    ylabel('SOC'); title('(b) Kalman-filter family', 'FontWeight','normal');
    if c == 1, legend({'Reference','EKF','SPKF','IEKF','SHEKF'}, 'Location','best'); end
    subplot(3, nC, 2*nC + c); hold on; grid on;
    plot(tt, 100*(YT(ii)-Pred_T(ii,8)), '-', 'Color', pal(8,:));
    plot(tt, 100*(YT(ii)-Pred_T(ii,2)), '-', 'Color', pal(2,:), 'LineWidth', 0.6);
    yline(0, 'k-', 'LineWidth', 0.4);
    xlabel('Time (min)'); ylabel('Error (% SOC)'); title('(c) Error: fusion and SPKF', 'FontWeight','normal');
    if c == 1, legend({'NNLS fusion','SPKF'}, 'Location','best'); end
end
saveFig(f, outDir, 'fig_tracking');
catch ME
    warning('Figure failed: %s', ME.message);
    try, close all; catch, end
end

try
% Fig 2: RMSE with bootstrap CI (log scale)
f = newFig(3.5, 2.8);
b = bar(100*M8(:,1), 'FaceColor','flat', 'EdgeColor','k', 'LineWidth', 0.4); b.CData = pal; hold on;
errorbar(1:8, 100*M8(:,1), 100*(M8(:,1)-ciR(1,:)'), 100*(ciR(2,:)'-M8(:,1)), 'k.', 'LineWidth', 0.6, 'CapSize', 3);
set(gca, 'YScale','log', 'XTick',1:8, 'XTickLabel', names8, 'XTickLabelRotation', 40);
ylabel('Test RMSE (% SOC)'); grid on;
saveFig(f, outDir, 'fig_rmse_ci');
catch ME
    warning('Figure failed: %s', ME.message);
    try, close all; catch, end
end

try
% Fig 3: accuracy vs latency (legend instead of text labels: no overlap)
f = newFig(3.5, 2.9); hold on; grid on;
mk = {'o','s','d','^','v','p','h','o'};  hh = gobjects(8,1);
for m = 1:8
    hh(m) = scatter(latAll(m), 100*M8(m,1), 38, pal(m,:), mk{m}, 'filled', 'MarkerEdgeColor','k', 'LineWidth', 0.4);
end
set(gca, 'XScale','log', 'YScale','log');
xlabel('Per-sample latency (ms)'); ylabel('Test RMSE (% SOC)');
legend(hh, names8, 'Location','northeastoutside', 'FontSize', 6);
saveFig(f, outDir, 'fig_accuracy_latency');
catch ME
    warning('Figure failed: %s', ME.message);
    try, close all; catch, end
end

try
% Fig 4: NNLS weights
f = newFig(3.5, 2.4);
bw = bar(w, 'FaceColor','flat', 'EdgeColor','k', 'LineWidth', 0.4); bw.CData = pal(1:7,:);
set(gca, 'XTick',1:7, 'XTickLabel', names7, 'XTickLabelRotation', 40); ylim([0 max(1, 1.05*max(w))]);
ylabel('NNLS weight'); grid on;
saveFig(f, outDir, 'fig_weights');
catch ME
    warning('Figure 4 failed: %s', ME.message); try, close all; catch, end
end

if ~isCross
try
% Fig 5: error correlation heat map (VAL)
f = newFig(3.5, 3.1);
imagesc(Ecorr, [-1 1]); colormap(gca, flipud(gray(256)));
cb = colorbar; cb.Label.String = 'Error correlation';
set(gca, 'XTick',1:7, 'XTickLabel', names7, 'YTick',1:7, 'YTickLabel', names7, 'XTickLabelRotation', 40);
axis square;
for i = 1:7, for j = 1:7
    text(j, i, sprintf('%.2f', Ecorr(i,j)), 'HorizontalAlignment','center', 'FontSize', 6, ...
        'Color', ternary(abs(Ecorr(i,j))>0.55,'w','k'));
end, end
saveFig(f, outDir, 'fig_error_correlation');
catch ME
    warning('Figure 5 failed: %s', ME.message); try, close all; catch, end
end

try
% Fig 6: fusion contribution analysis (linear axis starting at zero)
f = newFig(3.5, 2.8);
lab = abl(:,1);  bv = 100*ablRes(:,1);
bar(bv, 'FaceColor',[0.35 0.35 0.35], 'EdgeColor','k', 'LineWidth', 0.4); hold on;
for a = 1:numel(bv), text(a, bv(a), sprintf('%.2f', bv(a)), 'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize', 6); end
set(gca, 'XTick',1:numel(lab), 'XTickLabel', strrep(lab,' (7 estimators)',''), 'XTickLabelRotation', 40);
ylim([0 1.15*max(bv)]); ylabel('Test RMSE (% SOC)'); grid on;
saveFig(f, outDir, 'fig_ablation');
catch ME
    warning('Figure 6 failed: %s', ME.message); try, close all; catch, end
end

try
% Fig 7: physics-weight sweep (VAL and TEST)
f = newFig(3.5, 2.4);
tst = arrayfun(@(q) 100*rms(YT - predictPI(piNets{q}, X3_11_T, rc)), 1:numel(lamGrid));
bar([100*piValR(:), tst(:)]); legend({'VAL','TEST'}, 'Location','best');
set(gca, 'XTick',1:numel(lamGrid), 'XTickLabel', arrayfun(@(x) sprintf('%g',x), lamGrid, 'UniformOutput', false));
xlabel('Physics weight \lambda'); ylabel('PI-LSTM RMSE (% SOC)'); grid on;
saveFig(f, outDir, 'fig_lambda');
catch ME
    warning('Figure 7 failed: %s', ME.message); try, close all; catch, end
end
end

if numel(tempList) > 1
try
% Fig 8: RMSE versus temperature
f = newFig(3.5, 2.7); hold on; grid on;
[tl, so] = sort(tempList);
sel = [2 5 6 7 8];  lsx = {'-','-','-','-','--'};  mkx = {'o','s','^','v','d'};
for k = 1:numel(sel)
    plot(tl, 100*perTemp(so, sel(k)), lsx{k}, 'Color', pal(sel(k),:), 'Marker', mkx{k}, 'MarkerSize', 4, 'LineWidth', ternary(sel(k)==8,1.6,0.9));
end
leg = names8(sel);
if ~isempty(basePerTemp), plot(tl, 100*basePerTemp(so,1), ':', 'Color', [0.3 0.3 0.3], 'Marker', 'x', 'LineWidth', 1.0); leg{end+1} = 'LSTM'; end
set(gca, 'YScale','log'); xlabel('Temperature (\circC)'); ylabel('Test RMSE (% SOC)'); legend(leg, 'Location','northeast');
saveFig(f, outDir, 'fig_pertemp');
catch ME
    warning('Figure 8 failed: %s', ME.message); try, close all; catch, end
end
end

fprintf('\n=== ALL DONE. Everything is in ./%s ===\n', outDir);
fprintf('Upload the whole "%s" folder (or paste the console + results_claims.txt).\n', outDir);

%% ========================================================================
%  LOCAL FUNCTIONS -- DATA
%  ========================================================================
function [D, SEG, PAR] = loadBraunData(cfg)
    data = readtable(fullfile(cfg.dirBraun, 'Experimental_data_fresh_cell.csv'));
    ocvT = readtable(fullfile(cfg.dirBraun, 'OCV_vs_SOC_curve.csv'));
    t = data.Time;  N = numel(t);
    soc = refSOC(t, data.Current, data.Voltage, cfg.CN_Ah_braun);
    V = smoothdata(data.Voltage, 'movmean', 5);  I = smoothdata(data.Current, 'movmean', 5);
    iW0 = wltpStart(soc);  L = N - iW0 + 1;
    fprintf('  WLTP block starts at t = %.2f h (sample %d), %.1f h long\n', t(iW0)/3600, iW0, L/3600);
    switch cfg.protocol
        case 'regime',   nF = iW0 - 1;                  nV = nF + floor(0.35 * L);
        case 'wltpseen', nF = iW0 - 1 + floor(0.30*L);  nV = nF + floor(0.20 * L);
        otherwise, error('Braun protocol must be ''regime'' or ''wltpseen''.');
    end
    grp = [ones(nF,1); 2*ones(nV-nF,1); 3*ones(N-nV,1)];
    % Three segments (cycling / validation / test): every block is a new filter run that starts from an uncertain
    % initial SOC, as it would after a restart of the estimator. The record itself is not cut or shuffled.
    seg = grp;  pos = [(1:nF)'; (1:nV-nF)'; (1:N-nV)'];
    D = struct('V',V,'I',I,'T',data.Temperature,'soc',soc,'seg',seg,'pos',pos, ...
               'grp',grp,'par',ones(N,1),'tempC',nan(N,1),'cyc',zeros(N,1));
    ocvf = @(s) interp1(ocvT.SOC, ocvT.V0, max(0, min(1, s)), 'linear', 'extrap');
    [R0, R1, tau, rv] = identifyECM({V(1:nF)}, {I(1:nF)}, {soc(1:nF)}, ocvf, cfg.dt, [4e-3 1e-3 30; 1e-2 3e-3 100; 2e-3 1e-3 10]);
    PAR = struct('tempC', NaN, 'ocvG', ocvT.SOC, 'ocvV', ocvT.V0, 'dh', 0.001, 'R0', R0, 'R1', R1, ...
                 'tau', tau, 'Rn', rv, 'cap', cfg.CN_Ah_braun, 'Qn', diag([1e-10 1e-9]), 'allS', []);
    bnd = [1 nF+1 nV+1 N+1];  nm = {'fresh-fit','fresh-val','fresh-test'};
    SEG = struct('i0',{},'i1',{},'par',{},'soc0',{},'name',{},'tempC',{},'cyc',{},'grp',{},'cap',{});
    for a = 1:3
        SEG(a) = struct('i0',bnd(a),'i1',bnd(a+1)-1,'par',1,'soc0',soc(bnd(a)),'name',nm{a},'tempC',NaN,'cyc','WLTP','grp',a,'cap',cfg.CN_Ah_braun);
    end
end

function [D, SEG, PAR] = loadKollmeyerData(rootDir, cfg)
    cacheF = ['cache_data_' regexprep(rootDir, '\W', '_') '.mat'];
    if cfg.useCache && isfile(cacheF)
        S = load(cacheF);  SEGall = S.SEGall;  OCV = S.OCV;
        fprintf('  loaded cached 1 Hz data %s\n', cacheF);
    else
        [SEGall, OCV] = scanKollmeyer(rootDir);
        save(cacheF, 'SEGall', 'OCV', '-v7.3');
    end
    grp = assignGroups(SEGall, cfg.protocol, cfg.holdTemp, cfg.holdCycle);
    use = find(grp > 0)';
    nSeg = numel(use);  lens = arrayfun(@(k) numel(SEGall(k).V), use);  Ntot = sum(lens);
    D = struct('V',zeros(Ntot,1),'I',zeros(Ntot,1),'T',zeros(Ntot,1),'soc',zeros(Ntot,1),'seg',zeros(Ntot,1), ...
               'pos',zeros(Ntot,1),'grp',zeros(Ntot,1),'par',zeros(Ntot,1),'tempC',zeros(Ntot,1),'cyc',zeros(Ntot,1));
    cycCode = struct('UDDS',1,'HWFET',2,'LA92',3,'US06',4);
    % ---- one ECM parameter set per temperature that has FIT segments ----
    allT = [SEGall.tempC];
    fitTemps = unique(allT(grp' == 1));
    PAR = struct('tempC',{},'ocvG',{},'ocvV',{},'dh',{},'R0',{},'R1',{},'tau',{},'Rn',{},'cap',{},'Qn',{},'allS',{});
    for j = 1:numel(fitTemps)
        tc = fitTemps(j);  ii = find(grp' == 1 & allT == tc);  oi = find([OCV.tempC] == tc, 1);
        og = OCV(oi).g;  ov = OCV(oi).v;
        ocvf = @(s) interp1(og, ov, max(0, min(1, s)), 'linear', 'extrap');
        [R0, R1, tau, rv, allS] = identifyECM({SEGall(ii).V}, {SEGall(ii).I}, {SEGall(ii).soc}, ocvf, cfg.dt, [0.03 0.02 30; 0.05 0.05 40; 0.01 0.01 10; 0.1 0.1 100; 0.02 0.03 20; 0.08 0.08 60; 0.15 0.10 150; 0.005 0.005 15]);
        PAR(j) = struct('tempC', tc, 'ocvG', og, 'ocvV', ov, 'dh', 0.005, 'R0', R0, 'R1', R1, 'tau', tau, ...
                        'Rn', rv, 'cap', median([SEGall(ii).cap]), 'Qn', diag([1e-9 1e-8]), 'allS', allS);
    end
    % ---- temperatures without FIT data (held-out or cold-extrapolated): Arrhenius parameter model ----
    %  ln R0, ln R1, ln tau, ln Rn are linear in 1/T (Kelvin) over the identified temperatures, capacity is
    %  linear (or quadratic, from four temperatures) in T; the OCV curve of the nearest identified temperature is used.
    if ~strcmp(cfg.paramMode, 'nearest')
        newT = setdiff(unique(allT(grp' > 0)), fitTemps);
        if ~isempty(newT) && numel(fitTemps) >= 2
            iK = 1 ./ (fitTemps(:) + 273.15);
            lin = @(y, tq) exp(polyval(polyfit(iK, log(y(:)), 1), 1 ./ (tq + 273.15)));
            cDeg = 1 + (numel(fitTemps) >= 4);
            for q = 1:numel(newT)
                tq = newT(q);  [~, jn] = min(abs(fitTemps - tq));
                PAR(end+1) = struct('tempC', tq, 'ocvG', PAR(jn).ocvG, 'ocvV', PAR(jn).ocvV, 'dh', 0.005, ...
                    'R0', lin([PAR(1:numel(fitTemps)).R0], tq), 'R1', lin([PAR(1:numel(fitTemps)).R1], tq), ...
                    'tau', tauNew(cfg, [PAR(1:numel(fitTemps)).tau], lin, PAR(jn).tau, tq), ...
                    'Rn', lin([PAR(1:numel(fitTemps)).Rn], tq), ...
                    'cap', polyval(polyfit(fitTemps(:), [PAR(1:numel(fitTemps)).cap]', cDeg), tq), 'Qn', diag([1e-9 1e-8]), 'allS', []); %#ok<AGROW>
                fprintf('  Arrhenius parameters for unseen %g C: R0 %.4f, R1 %.4f, tau %.1f s, cap %.3f Ah (OCV of %g C)\n', ...
                    tq, PAR(end).R0, PAR(end).R1, PAR(end).tau, PAR(end).cap, fitTemps(jn));
            end
        end
    end
    % ---- concatenate used segments ----
    SEG = struct('i0',{},'i1',{},'par',{},'soc0',{},'name',{},'tempC',{},'cyc',{},'grp',{},'cap',{});
    p = 0;
    for a = 1:nSeg
        k = use(a);  n = lens(a);  ii = p + (1:n);
        [~, jp] = min(abs([PAR.tempC] - SEGall(k).tempC) - 1e-6 * [PAR.tempC]);   % nearest fitted temperature
        D.V(ii) = SEGall(k).V;  D.I(ii) = SEGall(k).I;  D.T(ii) = SEGall(k).T;  D.soc(ii) = SEGall(k).soc;
        D.seg(ii) = a;  D.pos(ii) = 1:n;  D.grp(ii) = grp(k);  D.par(ii) = jp;  D.tempC(ii) = SEGall(k).tempC;
        cy = SEGall(k).cyc;
        if isfield(cycCode, cy), D.cyc(ii) = cycCode.(cy); else, D.cyc(ii) = 5; end
        SEG(a) = struct('i0', p+1, 'i1', p+n, 'par', jp, 'soc0', SEGall(k).soc(1), 'name', SEGall(k).name, ...
                        'tempC', SEGall(k).tempC, 'cyc', cy, 'grp', grp(k), 'cap', SEGall(k).cap);
        p = p + n;
    end
end

function tq = tauNew(cfg, tauFit, lin, tauNear, tqC)
% Time constant of a temperature without identification data. 'arrhenius': ln tau linear in 1/T (clipped to [1, 600] s);
% 'nearest': tau of the nearest identified temperature; 'median': median over the identified temperatures.
    switch cfg.tauMode
        case 'nearest', tq = tauNear;
        case 'median',  tq = median(tauFit);
        otherwise,      tq = min(max(lin(tauFit, tqC), 1), 600);
    end
end

function [SEGall, OCV] = scanKollmeyer(rootDir)
    if ~isfolder(rootDir), error('Folder %s not found. Unzip the dataset next to this script.', rootDir); end
    d = dir(rootDir);  d = d([d.isdir] & ~startsWith({d.name}, '.'));
    SEGall = struct('V',{},'I',{},'T',{},'soc',{},'cap',{},'name',{},'tempC',{},'cyc',{},'depth',{});
    OCV = struct('tempC',{},'g',{},'v',{},'cap20',{});
    for a = 1:numel(d)
        tC = parseTemp(d(a).name);  if isnan(tC), continue; end
        fdir = fullfile(rootDir, d(a).name);
        c20 = dir(fullfile(fdir, '*C20DisCh.csv'));
        if isempty(c20), warning('No C20DisCh file in %s, folder skipped.', fdir); continue; end
        fprintf('  %s: reading C/20 test and drive cycles ...\n', d(a).name);
        [g, v, cap20] = ocvFromC20(fullfile(fdir, c20(1).name));
        OCV(end+1) = struct('tempC', tC, 'g', g, 'v', v, 'cap20', cap20); %#ok<AGROW>
        files = dir(fullfile(fdir, '*.csv'));
        tmp = SEGall([]);
        for q = 1:numel(files)
            tok = regexp(files(q).name, '_(UDDS|HWFET|LA92|US06|Mixed\d)\.csv$', 'tokens', 'once');
            if isempty(tok), continue; end
            [t, V, I, T, Ah] = readKollmeyerCSV(fullfile(fdir, files(q).name));
            kEnd = find(Ah == min(Ah), 1, 'first');                        % discharge ends at the Ah minimum
            t = t(1:kEnd); V = V(1:kEnd); I = I(1:kEnd); T = T(1:kEnd); Ah = Ah(1:kEnd);
            [V1, I1, T1, Ah1] = resample1s(t, V, I, T, Ah);
            depth = -min(Ah);  cap = depth / 0.95;
            tmp(end+1) = struct('V', V1, 'I', -I1, 'T', T1, 'soc', 1 + Ah1/cap, 'cap', cap, ...
                                'name', files(q).name, 'tempC', tC, 'cyc', tok{1}, 'depth', depth); %#ok<AGROW>
        end
        if isempty(tmp), continue; end
        keep = [tmp.depth] > 0.8 * median([tmp.depth]);
        for q = find(~keep), fprintf('    dropped incomplete file %s (%.2f Ah discharged)\n', tmp(q).name, tmp(q).depth); end
        SEGall = [SEGall, tmp(keep)]; %#ok<AGROW>
    end
    if isempty(SEGall), error('No drive-cycle files found in %s.', rootDir); end
end

function tC = parseTemp(name)
    s = regexprep(name, '^n', '-');  v = sscanf(s, '%d');
    if isempty(v), tC = NaN; else, tC = v(1); end
end

function grp = assignGroups(SEGall, protocol, holdTemp, holdCycle)
% 1 = FIT, 2 = VAL, 3 = TEST, 0 = unused. Mixed1..Mixed8 are random mixes of the four
% standard cycles; the last two Mixed files of every temperature are VAL.
% 'loco' : Mixed -> FIT/VAL, standard cycles -> TEST (all temperatures)
% 'tempx': temperatures >= 10 C train, colder temperatures are TEST (extrapolation)
% 'loto' : the temperature holdTemp is TEST, all others train (interpolation)
% 'sparse': like 'loto' with holdTemp = [25 0 -10]: ECM calibrated at only three temperatures
% 'lcyc' : leave one STANDARD cycle out (holdCycle): the Mixed files contain parts of all four standard cycles, so
%          they are not used at all; of the three remaining standard cycles the last one in the order
%          UDDS, HWFET, LA92, US06 is VAL and the other two are FIT; every file of holdCycle is TEST.
    n = numel(SEGall);  grp = zeros(n,1);  allT = [SEGall.tempC];
    for tc = unique(allT)
        idx = find(allT == tc);
        isMix = arrayfun(@(k) strncmp(SEGall(k).cyc, 'Mixed', 5), idx);
        mixIdx = idx(isMix);  stdIdx = idx(~isMix);
        fitIdx = []; valIdx = [];
        if ~isempty(mixIdx)
            [~, o] = sort(arrayfun(@(k) sscanf(SEGall(k).cyc, 'Mixed%d'), mixIdx));  mixIdx = mixIdx(o);
            nv = min(2, max(1, numel(mixIdx) - 1));
            valIdx = mixIdx(end-nv+1:end);  fitIdx = setdiff(mixIdx, valIdx, 'stable');
        end
        switch protocol
            case {'loco', 'cross'}
                grp(fitIdx) = 1;  grp(valIdx) = 2;  grp(stdIdx) = 3;
            case 'lcyc'
                stdNames = {'UDDS','HWFET','LA92','US06'};  rest = stdNames(~strcmp(stdNames, holdCycle));
                for k = stdIdx(:)'
                    if strcmp(SEGall(k).cyc, holdCycle), grp(k) = 3;
                    elseif strcmp(SEGall(k).cyc, rest{end}), grp(k) = 2;
                    else, grp(k) = 1; end
                end
            case 'tempx'
                if tc >= 10, grp(fitIdx) = 1;  grp(valIdx) = 2;
                else, grp(idx) = 3; end
            case {'loto', 'sparse'}   % leave temperature(s) out: every file at holdTemp is TEST
                if any(tc == holdTemp(:)), grp(idx) = 3;
                else, grp(fitIdx) = 1;  grp(valIdx) = 2;  grp(stdIdx) = 1; end
            otherwise
                error('Unknown protocol %s', protocol);
        end
    end
end

function [t, V, I, T, Ah] = readKollmeyerCSV(fn)
% Cycler CSV: header block, a line starting with 'Time Stamp', a units line, then rows
%   stamp,step,status,progtime,steptime,cycle,level,procedure,V,I,T,Ah,Wh,cnt,
% Raw sign: discharge current negative, Ah counter negative on discharge.
    fid = fopen(fn, 'r');  if fid < 0, error('Cannot open %s', fn); end
    nh = 0;
    while true
        ln = fgetl(fid);  nh = nh + 1;
        if ~ischar(ln), fclose(fid); error('No data header in %s', fn); end
        if strncmp(ln, 'Time Stamp', 10), hdr = ln; break; end
        if nh > 300, fclose(fid); error('No data header in %s', fn); end
    end
    fgetl(fid);                                                        % units line
    txt = fread(fid, inf, '*char')';  fclose(fid);
    nExtra = max(0, sum(hdr == ',') - 14);                             % extra numeric columns (e.g. IsSet2..5)
    fmt = ['%*[^,],%*d,%*[^,],%d:%d:%f,%*[^,],%*d,%*d,%*[^,],%f,%f,%f,%f,%*f,%*f,' repmat('%*f,', 1, nExtra)];
    v = sscanf(txt, fmt);
    nrec = floor(numel(v) / 7);
    v = reshape(v(1:7*nrec), 7, nrec)';
    t = v(:,1)*3600 + v(:,2)*60 + v(:,3);
    V = v(:,4);  I = v(:,5);  T = v(:,6);  Ah = v(:,7);
    ok = isfinite(t) & isfinite(V) & isfinite(I);
    t = t(ok); V = V(ok); I = I(ok); T = T(ok); Ah = Ah(ok);
    [t, iu] = unique(t, 'stable');  V = V(iu); I = I(iu); T = T(iu); Ah = Ah(iu);
end

function [V1, I1, T1, Ah1] = resample1s(t, V, I, T, Ah)
% Block-average to a uniform 1 s grid (gaps are filled by linear interpolation).
    t = t - t(1);  b = floor(t) + 1;  nb = max(b);
    cnt = accumarray(b, 1, [nb 1]);
    f = @(x) accumarray(b, x, [nb 1]) ./ max(cnt, 1);
    V1 = f(V); I1 = f(I); T1 = f(T); Ah1 = f(Ah);
    miss = cnt == 0;
    if any(miss)
        k = (1:nb)';
        V1(miss) = interp1(k(~miss), V1(~miss), k(miss), 'linear', 'extrap');
        I1(miss) = interp1(k(~miss), I1(~miss), k(miss), 'linear', 'extrap');
        T1(miss) = interp1(k(~miss), T1(~miss), k(miss), 'linear', 'extrap');
        Ah1(miss) = interp1(k(~miss), Ah1(~miss), k(miss), 'linear', 'extrap');
    end
end

function [g, ocv, cap20] = ocvFromC20(fn)
% OCV(SOC) = mean of the C/20 discharge and charge voltage curves (removes the
% small polarisation of either direction).
    [~, V, I, ~, Ah] = readKollmeyerCSV(fn);
    cap20 = -min(Ah);
    dis = I < -0.05;  chg = I > 0.05;
    sd = 1 + Ah(dis) / cap20;  vd = V(dis);
    sc = 1 + Ah(chg) / cap20;  vc = V(chg);
    [sd, iu] = unique(sd);  vd = vd(iu);  [sc, ju] = unique(sc);  vc = vc(ju);
    g = linspace(0, 1, 201)';
    od = interp1(sd, vd, g, 'linear', 'extrap');  oc = interp1(sc, vc, g, 'linear', 'extrap');
    ocv = 0.5 * (od + oc);
    ocv = cummax(movmean(ocv, 5)) + 1e-6 * g;                      % monotone, strictly increasing
end

function soc = refSOC(t, I, V, CN)
    % Coulomb counter with voltage-limit calibration (rule of the Braun et al. script)
    n = numel(t);  soc = zeros(n,1);
    for k = 2:n
        soc(k) = soc(k-1) - (t(k) - t(k-1)) / (CN*3600) * I(k);
        if (V(k) > 4.19 && abs(I(k)) < 4) || soc(k) > 1
            soc(k) = 1;
        elseif (V(k) < 3.01 && abs(I(k)) < 4) || soc(k) < 0
            soc(k) = 0;
        end
    end
end

function i0 = wltpStart(soc)
    % first full charge after 40 % of the record: end of the shallow cycling, start of WLTP
    n = numel(soc);
    i0 = find(soc >= 0.999 & (1:n)' > 0.4*n, 1, 'first');
end

%% ========================================================================
%  LOCAL FUNCTIONS -- MODEL, PRIORS, FEATURES
%  ========================================================================
function [R0, R1, tau, rv, allS] = identifyECM(Vc, Ic, Sc, ocvf, dt, P0)
% Least-squares identification of R0, R1, tau on terminal voltage. Nelder-Mead from several
% starting points (rows of P0); the best of them is kept. allS (one row per start) = [R0 R1 tau cost]
% so that the agreement between the starts can be reported.
    ocvc = cellfun(ocvf, Sc, 'UniformOutput', false);
    cost = @(p) ecmCost(p, Vc, Ic, ocvc, dt);
    opt = optimset('MaxIter', 1500, 'MaxFunEvals', 3000, 'TolX', 1e-7, 'TolFun', 1e-13, 'Display', 'off');
    bestC = inf;  bestP = log(P0(1,:));  allS = zeros(size(P0,1), 4);
    for q = 1:size(P0,1)
        [p, c] = fminsearch(cost, log(P0(q,:)), opt);
        [~, r0q, r1q, tq] = ecmCost(p, Vc, Ic, ocvc, dt);  allS(q,:) = [r0q r1q tq c];
        if c < bestC, bestC = c; bestP = p; end
    end
    [rv, R0, R1, tau] = ecmCost(bestP, Vc, Ic, ocvc, dt);
end

function ci = ecmBootstrap(Vc, Ic, Sc, ocvf, dt, B)
% Segment bootstrap of the identified parameters: resample the FIT segments of one temperature with
% replacement and refit (one start at the full-data optimum). Returns the 2.5 / 97.5 percentiles of
% [R0 R1 tau] as a 2x3 matrix.
    ns = numel(Vc);  out = zeros(B, 3);
    [R0, R1, tau] = identifyECM(Vc, Ic, Sc, ocvf, dt, [0.03 0.02 30]);
    for b = 1:B
        ix = 1 + floor(ns * mod(b * 0.6180339887 + (1:ns) * 0.7548776662, 1));        % deterministic resampling
        [r0, r1, ta] = identifyECM(Vc(ix), Ic(ix), Sc(ix), ocvf, dt, [R0 R1 tau]);
        out(b,:) = [r0 r1 ta];
    end
    ci = prctile(out, [2.5 97.5], 1);
end

function [c, R0, R1, tau] = ecmCost(p, Vc, Ic, ocvc, dt)
    R0 = exp(p(1));  R1 = exp(p(2));  tau = min(max(exp(p(3)), 1), 600);
    a = exp(-dt / tau);  sse = 0;  n = 0;
    for k = 1:numel(Vc)
        vrc = filter([0 R1*(1-a)], [1 -a], Ic{k});        % V_RC[k] = a V_RC[k-1] + R1 (1-a) I[k-1]
        r = ocvc{k} - vrc - R0 * Ic{k} - Vc{k};
        sse = sse + sum(r.^2);  n = n + numel(r);
    end
    c = sse / n;
end

function m = makeMdl(p, dt, o)
    if nargin < 3, o = struct('p0soc',1e-4,'ukfA',1,'ukfK',1); end
    m.Q = p.cap * 3600;  m.dt = dt;  g = p.ocvG;  v = p.ocvV;  hh = p.dh;
    m.ocv  = @(s) interp1(g, v, max(0, min(1, s)), 'linear', 'extrap');
    m.docv = @(s) (m.ocv(s + hh) - m.ocv(s - hh)) / (2*hh);
    m.R0 = p.R0;  m.R1 = p.R1;  m.tau = p.tau;
    m.alpha = exp(-dt / p.tau);
    m.A = [1, 0; 0, m.alpha];
    m.B = [-dt / m.Q; p.R1 * (1 - m.alpha)];
    m.h = @(x, I) m.ocv(x(1)) - x(2) - m.R0 * I;
    m.Qn = p.Qn;  m.Rn = p.Rn;  m.P0 = diag([o.p0soc, 1e-4]);  m.ukfA = o.ukfA;  m.ukfK = o.ukfK;
end

function [PR, tms, DG] = runPriors(V, I, SEG, segList, M, s0shift)
% Four Kalman filters on every listed segment. Each starts from SEG(k).s0, the initial SOC the estimator is given
% (the true initial SOC plus the deliberate error of cfg.initErr), plus an optional extra shift s0shift.
% DG.lmin / DG.nbad / DG.ncf (segment x filter): smallest eigenvalue of P, steps with lambda_min <= 0, Cholesky failures.
    PR = zeros(numel(V), 4);  tt = zeros(1,4);  steps = 0;
    DG = struct('lmin', nan(numel(SEG),4), 'nbad', zeros(numel(SEG),4), 'ncf', zeros(numel(SEG),4));
    for k = segList(:)'
        ii = SEG(k).i0 : SEG(k).i1;  m = M{SEG(k).par};
        s0 = min(max(SEG(k).s0 + s0shift, 0), 1);
        tic; [PR(ii,1), d1] = runEKF  (m, V(ii), I(ii), s0); tt(1) = tt(1) + toc;
        tic; [PR(ii,2), d2] = runUKF  (m, V(ii), I(ii), s0); tt(2) = tt(2) + toc;
        tic; [PR(ii,3), d3] = runIEKF (m, V(ii), I(ii), s0); tt(3) = tt(3) + toc;
        tic; [PR(ii,4), d4] = runSHEKF(m, V(ii), I(ii), s0); tt(4) = tt(4) + toc;
        dd = [d1; d2; d3; d4];  DG.lmin(k,:) = dd(:,1)';  DG.nbad(k,:) = dd(:,2)';  DG.ncf(k,:) = dd(:,3)';
        steps = steps + numel(ii) - 1;
    end
    tms = tt / max(steps, 1) * 1e3;
end

function [X8, X11] = makeFeatures(V, I, T, PR, segStart, baseIdx)
    dV = [0; diff(V)];  dI = [0; diff(I)];
    dV(segStart) = 0;   dI(segStart) = 0;                  % no differences across file boundaries
    dVdI = dV ./ (abs(dI) + 0.1);                          % regularised incremental resistance
    Pw = V .* I;
    if baseIdx >= 3, X8 = [V, I, T, dV, dI, dVdI, PR(:,1), PR(:,baseIdx)];   % IEKF/SHEKF base replaces SPKF as the 8th input
    else,            X8 = [V, I, T, dV, dI, dVdI, PR(:,1), PR(:,2)]; end
    X11 = [V, I, T, dV, dI, dVdI, Pw, PR(:,1), PR(:,2), PR(:,3), PR(:,4)];
end

function out = evalScenario(ctx, Vp, Ip, Tp, s0shift)
% Re-runs filters, features and every trained model on (possibly perturbed) data of the
% TEST windows ctx.eT. Nothing is refitted.
    W = ctx.cfg.W;  e = ctx.eT;  D = ctx.D;
    segT = unique(D.seg(e))';
    PRp = runPriors(Vp, Ip, ctx.SEG, segT, ctx.M, s0shift);
    [X8p, X11p] = makeFeatures(Vp, Ip, Tp, PRp, [ctx.SEG.i0]', ctx.baseIdx);
    X8p = (X8p - ctx.mu8) ./ ctx.sig8;   X11p = (X11p - ctx.mu11) ./ ctx.sig11;
    pb = predict(ctx.net_bi, toCell(seq3(X8p, e, W)), 'MiniBatchSize', 2048);
    pp = predictPI(ctx.net_pi, seq3(X11p, e, W), ctx.rc);
    po = predict(ctx.mdl_boost, X8p(e,:));
    res = double(ctx.cfg.residual);
    prior = PRp(e, ctx.baseIdx);
    Pb = min(max(res*prior + double(pb(:)) / ctx.resS, 0), 1);
    Po = min(max(res*prior + double(po(:)), 0), 1);
    Phi = double([PRp(e,1:4), Po, pp(:), Pb]);
    F = min(max(Phi * ctx.w, 0), 1);
    Pr = [Phi, F];  Y = D.soc(e);
    out.rmse = zeros(1,8); out.mae = zeros(1,8); out.max = zeros(1,8);
    for m = 1:8
        mm = metrics(Y, Pr(:,m));  out.rmse(m) = mm(1);  out.mae(m) = mm(2);  out.max(m) = mm(3);
    end
end

%% LOCAL FUNCTIONS -- KALMAN FILTERS
function l = lmin2(P)
% smallest eigenvalue of a symmetric 2x2 matrix (closed form, cheap enough to log at every step)
    a = P(1,1);  d = P(2,2);  b = 0.5*(P(1,2) + P(2,1));
    l = 0.5*(a + d) - sqrt(0.25*(a - d)^2 + b^2);
end

function [soc, dg] = runEKF(m, V, I, s0)
% dg = [min eigenvalue of P over the run, number of steps with lambda_min <= 0, Cholesky failures (0 for the EKF)]
    N = numel(V); soc = zeros(N,1); x = [s0; 0]; P = m.P0; soc(1) = s0;  lmin = inf;  nbad = 0;
    for k = 2:N
        Ik = I(k-1);
        xp = m.A*x + m.B*Ik;  xp(1) = min(max(xp(1),0),1);
        Pp = m.A*P*m.A' + m.Qn;
        H  = [m.docv(xp(1)), -1];
        S  = H*Pp*H' + m.Rn;  K = Pp*H'/S;
        x  = xp + K*(V(k) - m.h(xp, Ik));
        P  = (eye(2)-K*H)*Pp*(eye(2)-K*H)' + K*m.Rn*K';
        x(1) = min(max(x(1),0),1);  soc(k) = x(1);
        l = lmin2(P);  lmin = min(lmin, l);  nbad = nbad + (l <= 0);
    end
    dg = [lmin, nbad, 0];
end

function [soc, dg] = runUKF(m, V, I, s0)
% Unscented (sigma-point) Kalman filter. Scaling: lambda = a^2 (n + ka) - n.
% m.ukfA / m.ukfK set a and ka. The default (a = 1, ka = 1, so lambda = 1) keeps the weights O(1).
% The legacy setting a = 1e-3, ka = 0 gives lambda ~ -2 and a zeroth covariance weight of about -1e6,
% which lets P lose positive definiteness (see the SPKF diagnostic in the pipeline).
    N = numel(V); n = 2; soc = zeros(N,1); x = [s0; 0]; P = m.P0; soc(1) = s0;
    a = m.ukfA; be = 2; ka = m.ukfK; lam = a^2*(n+ka) - n;
    Wm = [lam/(n+lam); repmat(1/(2*(n+lam)), 2*n, 1)];
    Wc = Wm;  Wc(1) = Wc(1) + (1 - a^2 + be);
    lmin = inf;  nbad = 0;  ncf = 0;
    for k = 2:N
        Ik = I(k-1);
        [R, p] = chol((n+lam)*P, 'lower');
        if p > 0
            ncf = ncf + 1;                                   % P was not positive definite: log it, then regularise
            P = (P+P')/2 + 1e-12*eye(n); R = chol((n+lam)*P, 'lower');
        end
        chi = [x, x + R, x - R];
        for j = 1:2*n+1
            chi(:,j) = m.A*chi(:,j) + m.B*Ik;  chi(1,j) = min(max(chi(1,j),0),1);
        end
        xp = chi*Wm;  Pp = m.Qn;
        for j = 1:2*n+1, d = chi(:,j)-xp; Pp = Pp + Wc(j)*(d*d'); end
        g = zeros(1,2*n+1);
        for j = 1:2*n+1, g(j) = m.h(chi(:,j), Ik); end
        yp = g*Wm;  Pyy = m.Rn;  Pxy = zeros(n,1);
        for j = 1:2*n+1
            dy = g(j)-yp;  dx = chi(:,j)-xp;
            Pyy = Pyy + Wc(j)*dy*dy;  Pxy = Pxy + Wc(j)*dx*dy;
        end
        K = Pxy/Pyy;  x = xp + K*(V(k)-yp);  P = Pp - K*Pyy*K';
        x(1) = min(max(x(1),0),1);  soc(k) = x(1);
        l = lmin2(P);  lmin = min(lmin, l);  nbad = nbad + (l <= 0);
    end
    dg = [lmin, nbad, ncf];
end

function [soc, dg] = runIEKF(m, V, I, s0)
    N = numel(V); soc = zeros(N,1); x = [s0; 0]; P = m.P0; soc(1) = s0;  lmin = inf;  nbad = 0;
    for k = 2:N
        Ik = I(k-1);
        xp = m.A*x + m.B*Ik;  xp(1) = min(max(xp(1),0),1);
        Pp = m.A*P*m.A' + m.Qn;
        xi = xp;
        for it = 1:5
            H = [m.docv(xi(1)), -1];
            e = V(k) - m.h(xi, Ik) - H*(xp - xi);
            S = H*Pp*H' + m.Rn;  K = Pp*H'/S;
            xn = xp + K*e;  xn(1) = min(max(xn(1),0),1);
            done = norm(xn - xi) < 1e-6;  xi = xn;
            if done, break; end
        end
        x = xi;
        H = [m.docv(x(1)), -1];  S = H*Pp*H' + m.Rn;  K = Pp*H'/S;
        P = (eye(2)-K*H)*Pp*(eye(2)-K*H)' + K*m.Rn*K';
        soc(k) = x(1);
        l = lmin2(P);  lmin = min(lmin, l);  nbad = nbad + (l <= 0);
    end
    dg = [lmin, nbad, 0];
end
function [soc, dg] = runSHEKF(m, V, I, s0)
% Sage-Husa adaptive EKF: the measurement-noise variance is re-estimated online from
% the innovation (forgetting factor b); the process noise stays fixed. The estimate is
% bounded to [0.2, 20] x the identified value so the adaptation cannot run away (the
% earlier version that also adapted Q and a measurement bias diverged on the Braun data).
    N = numel(V); soc = zeros(N,1); x = [s0; 0]; P = m.P0; soc(1) = s0;  lmin = inf;  nbad = 0;
    Rs = m.Rn; Rlo = 0.2*m.Rn; Rhi = 20*m.Rn; b = 0.98;
    for k = 2:N
        Ik = I(k-1);
        xp = m.A*x + m.B*Ik;  xp(1) = min(max(xp(1),0),1);
        Pp = m.A*P*m.A' + m.Qn;
        H  = [m.docv(xp(1)), -1];
        ye = V(k) - m.h(xp, Ik);
        S  = H*Pp*H' + Rs;  K = Pp*H'/S;
        x  = xp + K*ye;
        P  = (eye(2)-K*H)*Pp*(eye(2)-K*H)' + K*Rs*K';
        x(1) = min(max(x(1),0),1);
        dk = (1-b)/(1-b^k);
        Rs = min(max((1-dk)*Rs + dk*(ye^2 - H*Pp*H'), Rlo), Rhi);
        soc(k) = x(1);
        l = lmin2(P);  lmin = min(lmin, l);  nbad = nbad + (l <= 0);
    end
    dg = [lmin, nbad, 0];
end

%% ========================================================================
%  LOCAL FUNCTIONS -- UTILITIES
%  ========================================================================
function r = rms(x), r = sqrt(mean(x.^2, 'all')); end
function y = clip01(x), y = min(max(x, 0), 1); end
function y = ternary(c, a, b), if c, y = a; else, y = b; end, end
function i = argmin(x), [~, i] = min(x); end
function i = argmax(x), [~, i] = max(x); end

function s = pstr(p)
    if p < 1e-4, s = '$<10^{-4}$'; else, s = sprintf('%.4f', p); end
end

function s = texName(n)
    s = strrep(n, '_', '\_');
end

function m = metrics(y, yh)
    e = y - yh;
    m = [sqrt(mean(e.^2)), mean(abs(e)), max(abs(e)), 1 - sum(e.^2)/sum((y-mean(y)).^2)];
end

function [Xn, mu, sg] = zscoreTrain(X, idx)
    mu = mean(X(idx,:), 1);  sg = std(X(idx,:), 0, 1);
    sg(sg < 1e-10) = 1;  Xn = (X - mu) ./ sg;
end

function X3 = seq3(Xn, ends, W)
    n = numel(ends); C = size(Xn, 2);
    X3 = zeros(C, W, n, 'single');
    for j = 1:n, X3(:,:,j) = single(Xn(ends(j)-W+1:ends(j), :))'; end
end

function c = toCell(X3)
    if ndims(X3) == 2, c = {X3}; return; end
    c = squeeze(num2cell(X3, [1 2]));
    c = c(:);
end

function obj = loadOrTrain(file, sig, useCache, trainFcn)
    if useCache && isfile(file)
        S = load(file);
        if isfield(S,'sig') && isequal(S.sig, sig)
            fprintf('  Loaded cached model %s\n', file); obj = S.obj; return;
        end
    end
    obj = trainFcn();
    save(file, 'obj', 'sig');
end

function t = medianTime(fcn, n)
    fcn();                                   % warm-up
    ts = zeros(n,1);
    for i = 1:n, tic; fcn(); ts(i) = toc; end
    t = median(ts) * 1e3;
end

function w = nnlsSimplex(Phi, y)
% Least squares on the probability simplex (w >= 0, sum(w) = 1), solved on the
% normal equations by accelerated projected gradient (FISTA).
    Phi = double(Phi); y = double(y(:)); m = size(Phi,1);
    G = (Phi' * Phi) / m;  c = (Phi' * y) / m;  n = numel(c);
    L = max(eig((G + G')/2));
    w = ones(n,1)/n;  z = w;  t = 1;
    for it = 1:300000
        wn = projSimplex(z - (G*z - c)/L);
        tn = (1 + sqrt(1 + 4*t^2))/2;
        z  = wn + ((t-1)/tn) * (wn - w);
        if norm(wn - w, 1) < 1e-13, w = wn; break; end
        w = wn;  t = tn;
    end
    w = max(w, 0);  w = w / sum(w);
end

function x = projSimplex(v)
    n = numel(v);  u = sort(v, 'descend');  css = cumsum(u);
    rho = find(u .* (1:n)' > (css - 1), 1, 'last');
    theta = (css(rho) - 1) / rho;
    x = max(v - theta, 0);
end

function [ci, boots] = blockBoot(SE, B, L)
    n = size(SE,1); nb = ceil(n / L); boots = zeros(B, size(SE,2));
    for b = 1:B
        st  = randi(n - L + 1, 1, nb);
        idx = reshape(st + (0:L-1)', [], 1);  idx = idx(1:n);
        boots(b,:) = sqrt(mean(SE(idx,:), 1));
    end
    ci = prctile(boots, [2.5 97.5], 1);
end

function f = newFig(wIn, hIn)
    f = figure('Color','w','Units','inches','Position',[1 1 wIn hIn], ...
               'PaperUnits','inches','PaperSize',[wIn hIn],'PaperPosition',[0 0 wIn hIn]);
end

function saveFig(f, outDir, name)
    base = fullfile(outDir, name);
    ax = findall(f, 'Type', 'axes');
    for q = 1:numel(ax)                          % remove the interactive axes toolbar before export
        try, ax(q).Toolbar.Visible = 'off'; catch, end
        try, disableDefaultInteractivity(ax(q)); catch, end
    end
    try, exportgraphics(f, [base '.pdf'], 'ContentType','vector');
    catch, print(f, [base '.pdf'], '-dpdf', '-painters'); end
    print(f, [base '.png'], '-dpng', '-r600');
    print(f, [base '.tif'], '-dtiff', '-r600');
    close(f);
end

%% ========================================================================
%  LOCAL FUNCTIONS -- NETWORKS
%  ========================================================================
function opts = seqOptions(Xva, Yva, epochs)
    opts = trainingOptions('adam', 'MaxEpochs', epochs, 'MiniBatchSize', 256, ...
        'InitialLearnRate', 1e-3, 'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', max(1, round(0.4*epochs)), 'LearnRateDropFactor', 0.5, ...
        'GradientThreshold', 1, 'L2Regularization', 1e-4, 'Shuffle', 'every-epoch', ...
        'ValidationData', {Xva, Yva(:)}, 'ValidationFrequency', 100, 'ValidationPatience', 8, ...
        'OutputNetwork', 'best-validation-loss', 'Verbose', true, 'VerboseFrequency', 200, ...
        'Plots', 'none');
end

function net = trainBiLSTM(Xtr, Ytr, Xva, Yva, nfeat, epochs)
    layers = [
        sequenceInputLayer(nfeat, 'Name','input')
        bilstmLayer(128, 'OutputMode','last', 'Name','bilstm1')
        dropoutLayer(0.2, 'Name','drop1')
        fullyConnectedLayer(64, 'Name','fc1')
        reluLayer('Name','relu1')
        dropoutLayer(0.1, 'Name','drop2')
        fullyConnectedLayer(32, 'Name','fc2')
        reluLayer('Name','relu2')
        fullyConnectedLayer(1, 'Name','fc_out')
        regressionLayer('Name','output')];
    net = trainNetwork(Xtr, Ytr(:), layers, seqOptions(Xva, Yva, epochs));
end

function net = trainSeqBaseline(kind, Xtr, Ytr, Xva, Yva, nfeat, epochs)
% Literature-style baselines on the measured inputs only (V, I, T and their differences).
    switch kind
        case 'LSTM',     core = lstmLayer(128, 'OutputMode','last', 'Name','rnn');
        case 'GRU',      core = gruLayer(128, 'OutputMode','last', 'Name','rnn');
        case 'CNN-LSTM', core = [convolution1dLayer(5, 32, 'Padding','same', 'Name','conv'); ...
                                 reluLayer('Name','relu0'); lstmLayer(128, 'OutputMode','last', 'Name','rnn')];
        otherwise, error('unknown baseline %s', kind);
    end
    layers = [
        sequenceInputLayer(nfeat, 'Name','input')
        core
        dropoutLayer(0.2, 'Name','drop1')
        fullyConnectedLayer(64, 'Name','fc1')
        reluLayer('Name','relu1')
        fullyConnectedLayer(1, 'Name','fc_out')
        regressionLayer('Name','output')];
    net = trainNetwork(Xtr, Ytr(:), layers, seqOptions(Xva, Yva, epochs));
end

function P = trainPILSTM(X3tr, Ytr, Qtr, X3va, Yva, W, dt, muI, sigI, lambda, epochs, batch, rc) %#ok<INUSL>
% Physics-informed unidirectional LSTM written with the functional API
% (lstm / fullyconnect) so that the layout is explicit: C x B x T.
% rc.col > 0 : the network outputs a correction that is added to the SPKF prior
% (column rc.col of the z-scored input), so the predicted SOC trajectory is
%     SOC_t = SPKF_t + out_t / rc.scale ;
% the loss is   MSE(SOC_T, reference)  +  lambda * MSE( dSOC_t , -I_t dt / Q )
% where Q is the capacity (coulombs) of the cell/temperature of that window.
    nf = size(X3tr,1);  H = 128;  H1 = 32;
    P = initPI(nf, H, H1, rc.col > 0);
    Ytr = single(Ytr(:));  Qtr = single(Qtr(:));  nObs = size(X3tr,3);  nIter = floor(nObs/batch);
    avg = []; avgSq = []; it = 0; best = P; bestR = inf;
    dropEvery = max(1, round(0.4*epochs));
    for ep = 1:epochs
        lr = 1e-3 * 0.5^floor((ep-1)/dropEvery);
        perm = randperm(nObs);
        for k = 1:nIter
            it = it + 1;  bi = perm((k-1)*batch+1 : k*batch);
            Xb = permute(X3tr(:,:,bi), [1 3 2]);          % C x B x T
            dlX = dlarray(Xb, 'CBT');
            dlY = dlarray(Ytr(bi)');                      % 1 x B
            Qb  = Qtr(bi)';                               % 1 x B (coulombs)
            Eb  = -((Xb(2, :, 1:end-1) * sigI + muI) ./ Qb) * dt;   % Coulomb-counting SOC increments, 1 x B x (T-1)
            dlE = dlarray(single(Eb));
            [loss, g, lD, lP] = dlfeval(@piGradients, P, dlX, dlY, dlE, lambda, rc);
            g = dlupdate(@(z) max(min(z,1),-1), g);
            [P, avg, avgSq] = adamupdate(P, g, avg, avgSq, it, lr);
        end
        yv = predictPI(P, X3va, rc);  r = sqrt(mean((yv - Yva(:)).^2));
        fprintf('  PI-LSTM epoch %2d/%d | loss %.6f | data %.6f | phys %.3e | val RMSE %.5f\n', ...
            ep, epochs, double(extractdata(loss)), double(extractdata(lD)), ...
            double(extractdata(lP)), r);
        if r < bestR, bestR = r; best = P; end
    end
    P = best;
end

function P = initPI(nf, H, H1, smallOut)
    glorot = @(r,c) dlarray(single((2*rand(r,c)-1) * sqrt(6/(r+c))));
    P.W  = glorot(4*H, nf);          % input weights
    P.R  = glorot(4*H, H);           % recurrent weights
    b = zeros(4*H,1,'single'); b(H+1:2*H) = 1;      % forget-gate bias = 1
    P.b  = dlarray(b);
    P.W1 = glorot(H1, H);   P.b1 = dlarray(zeros(H1,1,'single'));
    P.W2 = glorot(1, H1);   P.b2 = dlarray(zeros(1,1,'single'));
    if smallOut, P.W2 = 0.01 * P.W2; end             % start close to the Kalman prior
end

function out = piForward(P, dlX, training)
% dlX: C x B x T labelled 'CBT'. Returns 1 x B x T (unlabelled).
    B = size(dlX, 2);  H = size(P.R, 2);
    h0 = dlarray(zeros(H, B, 'single'));  c0 = h0;
    Y = lstm(dlX, h0, c0, P.W, P.R, P.b);           % H x B x T
    if training
        [Hh, Bb, ~] = size(Y);
        keep = single(rand(Hh, Bb) > 0.2) / 0.8;     % inverted dropout, one mask shared over time
        Y = dlarray(stripdims(Y) .* keep, 'CBT');
    end
    Z = relu(fullyconnect(Y, P.W1, P.b1));          % H1 x B x T
    O = fullyconnect(Z, P.W2, P.b2);                % 1 x B x T
    out = stripdims(O);
end

function s = piSOC(out, dlX, rc)
% SOC trajectory (1 x B x T) from the network output.
    if rc.col > 0
        Xs = stripdims(dlX);
        s = Xs(rc.col, :, :) * rc.sg + rc.mu + out / rc.scale;
    else
        s = out;
    end
end

function [loss, gradients, lossData, lossPhys] = piGradients(P, dlX, dlY, dlE, lambda, rc)
    out  = piForward(P, dlX, true);                 % 1 x B x T
    soc  = piSOC(out, dlX, rc);                     % 1 x B x T
    yFin = soc(1, :, end);                          % 1 x B
    lossData = mean((yFin(:) - dlY(:)).^2);
    dS   = soc(1, :, 2:end) - soc(1, :, 1:end-1);   % predicted SOC increments
    lossPhys = mean((dS - dlE).^2, 'all');          % deviation from Coulomb counting (Q of that window's cell/temperature)
    L2 = 0; f = fieldnames(P);
    for q = 1:numel(f), L2 = L2 + sum(P.(f{q}).^2, 'all'); end
    loss = lossData + lambda*lossPhys + 1e-4*L2;
    gradients = dlgradient(loss, P);
end

function y = predictPI(P, X3, rc)
    n = size(X3,3); y = zeros(n,1); cs = 2048;
    for s = 1:cs:n
        e = min(s+cs-1, n);
        dlX = dlarray(permute(X3(:,:,s:e), [1 3 2]), 'CBT');
        o = extractdata(piSOC(piForward(P, dlX, false), dlX, rc));   % 1 x B x T
        y(s:e) = double(reshape(o(1, :, end), [], 1));
    end
    y = min(max(y, 0), 1);
end

function y = netKF(z, I, Qc, seg, R, q, dt)
% Scalar Kalman filter: state SOC, Coulomb-counting prediction, network output z as pseudo-measurement.
% z, I, Qc (coulombs), seg are aligned column vectors of the test windows; filter restarts when seg changes.
    n = numel(z);  y = zeros(n,1);
    for k = 1:n
        if k == 1 || seg(k) ~= seg(k-1)
            s = z(k);  P = R;
        else
            s = s - I(k-1) * dt / Qc(k-1);  P = P + q;
            K = P / (P + R);  s = s + K * (z(k) - s);  P = (1 - K) * P;
        end
        y(k) = min(max(s, 0), 1);
    end
end

%% ========================================================================
%  LOCAL FUNCTIONS -- v3.3 STATISTICS AND EVIDENCE BLOCKS
%  ========================================================================
function [ci, boots, uRMSE, uN] = segBoot(SE, key, B)
% Bootstrap over whole units (segments): resample the units with replacement and recompute the pooled RMSE.
    [~, ~, g] = unique(key);  nU = max(g);  nM = size(SE, 2);
    uS = zeros(nU, nM);  uN = zeros(nU, 1);
    for u = 1:nU, ii = (g == u);  uS(u,:) = sum(SE(ii,:), 1);  uN(u) = sum(ii); end
    uRMSE = sqrt(uS ./ uN);
    boots = zeros(B, nM);
    for b = 1:B
        ix = randi(nU, nU, 1);
        boots(b,:) = sqrt(sum(uS(ix,:), 1) / sum(uN(ix)));
    end
    ci = prctile(boots, [2.5 97.5], 1);
end

function s = ccRun(I, Qc, SEG, s0v, segList, gain, off, dt, N)
% Coulomb counting per segment from the initial SOC s0v(k): SOC falls by (gain*I + off)*dt/Qc each step (discharge positive).
    s = zeros(N, 1);
    for k = segList(:)'
        ii = SEG(k).i0 : SEG(k).i1;
        inc = ((1 + gain) * I(ii) + off) * dt ./ Qc(ii);
        s(ii) = s0v(k) - [0; cumsum(inc(1:end-1))];
    end
    s = min(max(s, 0), 1);
end

function [medT, frac] = convStats(AE, seg, pos, tol, hold)
% First time (seconds from the start of the segment) after which the absolute error stays inside tol for hold samples.
% frac = share of segments that do converge; medT = median convergence time over the segments that do.
    us = unique(seg);  nM = size(AE, 2);  tc = nan(numel(us), nM);
    for a = 1:numel(us)
        ii = find(seg == us(a));  p = pos(ii);
        for m = 1:nM
            ok = double(AE(ii,m) <= tol);
            if numel(ok) < hold, continue; end
            c = conv(ok, ones(hold,1), 'valid');
            j = find(c >= hold, 1, 'first');
            if ~isempty(j), tc(a,m) = p(j); end
        end
    end
    frac = mean(~isnan(tc), 1);
    medT = nan(1, nM);
    for m = 1:nM, v = tc(~isnan(tc(:,m)), m); if ~isempty(v), medT(m) = median(v); end, end
end

function net = trainBiLSTMs(seed, varargin)
    rng(seed);  net = trainBiLSTM(varargin{:});
end

function P = trainPILSTMs(seed, varargin)
    rng(seed);  P = trainPILSTM(varargin{:});
end

function p = setf(p, f, v), p.(f) = v; end

function O = paramOracle(D, SEG, PAR, M, cfg, mo, rootDir) %#ok<INUSL>
% Filters only. For every TEST temperature that has an Arrhenius (extrapolated) parameter set, replace one parameter of
% that set at a time by the value identified directly on that temperature's own data. This is an oracle (it uses the
% test-temperature data) and a diagnostic, not a method. The same routine tests other laws for the time constant.
    O = struct('ok', false, 'names', {{}}, 'R', [], 'temps', [], 'prm', [], 'have', []);
    cacheF = ['cache_data_' regexprep(rootDir, '\W', '_') '.mat'];
    if ~isfile(cacheF), fprintf('  oracle skipped: %s not found\n', cacheF); return; end
    S = load(cacheF);  OCV = S.OCV;
    isTest = [SEG.grp] == 3;  tq = unique([SEG(isTest).tempC]);  tq = tq(~isnan(tq));
    isFit = arrayfun(@(p) ~isempty(p.allS), PAR);   fitIdx = find(isFit);
    vnames = {'extrapolated (as used)', 'R0 direct', 'R1 direct', 'tau direct', 'Q direct', 'OCV direct', ...
              'R0, R1, tau direct', 'all direct (incl. Rn)', 'tau = nearest identified', 'tau = median identified'};
    filt = {@runEKF, @runUKF, @runIEKF, @runSHEKF};
    R = nan(numel(vnames), 4, numel(tq));  prm = nan(numel(tq), 9);  have = false(1, numel(tq));
    starts = [0.03 0.02 30; 0.05 0.05 40; 0.01 0.01 10; 0.1 0.1 100; 0.02 0.03 20; 0.08 0.08 60; 0.15 0.10 150; 0.005 0.005 15];
    for a = 1:numel(tq)
        j = find([PAR.tempC] == tq(a) & ~isFit, 1);                 % the extrapolated set of this temperature
        if isempty(j), continue; end
        segs = find(isTest & [SEG.tempC] == tq(a));
        oi = find([OCV.tempC] == tq(a), 1);  if isempty(oi) || isempty(segs), continue; end
        og = OCV(oi).g;  ov = OCV(oi).v;
        ocvf = @(s) interp1(og, ov, max(0, min(1, s)), 'linear', 'extrap');
        Vc = cell(1, numel(segs));  Ic = Vc;  Sc = Vc;
        for q = 1:numel(segs), ii = SEG(segs(q)).i0:SEG(segs(q)).i1;  Vc{q} = D.V(ii);  Ic{q} = D.I(ii);  Sc{q} = D.soc(ii); end
        [R0d, R1d, taud, rvd] = identifyECM(Vc, Ic, Sc, ocvf, cfg.dt, starts);
        capd = median([SEG(segs).cap]);
        Pe = PAR(j);
        [~, jn] = min(abs([PAR(fitIdx).tempC] - tq(a)));  jn = fitIdx(jn);
        Vr = cell(1, numel(vnames));
        Vr{1} = Pe;
        Vr{2} = setf(Pe, 'R0', R0d);  Vr{3} = setf(Pe, 'R1', R1d);  Vr{4} = setf(Pe, 'tau', taud);  Vr{5} = setf(Pe, 'cap', capd);
        Vr{6} = setf(setf(Pe, 'ocvG', og), 'ocvV', ov);
        Vr{7} = setf(setf(setf(Pe, 'R0', R0d), 'R1', R1d), 'tau', taud);
        Pall = Vr{7};  Pall.cap = capd;  Pall.ocvG = og;  Pall.ocvV = ov;  Pall.Rn = rvd;  Vr{8} = Pall;
        Vr{9} = setf(Pe, 'tau', PAR(jn).tau);  Vr{10} = setf(Pe, 'tau', median([PAR(fitIdx).tau]));
        for v = 1:numel(Vr)
            mv = makeMdl(Vr{v}, cfg.dt, mo);
            for f = 1:4
                se = 0;  n = 0;
                for q = 1:numel(segs)
                    k = segs(q);  ii = SEG(k).i0:SEG(k).i1;
                    s = filt{f}(mv, D.V(ii), D.I(ii), SEG(k).s0);
                    e = D.soc(ii) - s;  e = e(D.pos(ii) >= cfg.W);  se = se + sum(e.^2);  n = n + numel(e);
                end
                R(v, f, a) = sqrt(se / n);
            end
        end
        prm(a,:) = [tq(a), Pe.R0, R0d, Pe.R1, R1d, Pe.tau, taud, Pe.cap, capd];  have(a) = true;
    end
    O.ok = any(have);  O.names = vnames;  O.R = R;  O.temps = tq;  O.prm = prm;  O.have = have;
end

function out = colfun(X, f)
% apply f to the non-NaN entries of every column of X (NaN if a column has none)
    out = nan(1, size(X, 2));
    for j = 1:size(X, 2), v = X(~isnan(X(:,j)), j);  if ~isempty(v), out(j) = f(v); end, end
end
