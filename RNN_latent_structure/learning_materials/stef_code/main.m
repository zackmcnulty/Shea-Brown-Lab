feature('setprecision', 24)

%% Create Environment, Agent and Data
% for i_sim = 26:100
i_sim = 2;
% create environment from figure
env_name = 'openmaze64';
walls = ~rgb2gray(imread(['data/environments/' env_name '.png']));
correlation = 1;
env = environment(walls, correlation);

% create agent and set folders to save data.
age = agent_net([size(walls,1),size(walls,2)]);
age.path2navigation = 'data/navigation/';
age.path2train = 'data/train/';
  
% do random exploration
N_steps = 500000;
rng(i_sim)
age.N_sensors = 40;
age.angle_sensors = linspace(-pi/4,pi/4,age.N_sensors);
age.N_o = age.N_sensors*4;
age.o = zeros(1,age.N_o);  
age.hist.o = zeros(1,age.N_o);  
age = age.explore_randomly(env, N_steps, true);

% save the data
% dataname = ['data_' env_name, '_seed' num2str(i_sim), '_Nh' num2str(age.)];       
dataname = ['data_' env_name, '_corr ', num2str(correlation), '_Nh' num2str(age.N_h), '_Nsensors', num2str(age.N_sensors)];       
age = age.save2train(N_steps, dataname);

% after you run it once then you can load the file;
% age = age.load2agent(dataname,false); 

% age.weights.rec_weight = RandOrthMat(age.N_h);
% age.weights.rec_weight = rand(age.N_h)/250;
% dataname = ['data_' env_name, '_Nh' num2str(age.N_h) '_randW'];       
% age = age.save2train(N_steps, dataname);

%% Train the neural network of the agent in python
N_epochs = 400;
age.sparsity = 0;
simulation_name = [env_name '_sparsity' num2str(age.sparsity) '_seed' num2str(i_sim) '_orthW_pred_Nh100'];
filesave = [age.path2train simulation_name 'nopred'];
filesavenp = [age.path2train simulation_name 'pred'];

%% Now run the python script for training saving the weights at the end of each epoch
system(['python net.py --epochs ' num2str(N_epochs) ...                                     %number of epochs to train
        ' --epoch_ini ' num2str(age.last_epoch) ...                                         %initial epoch
        ' --N_h ' num2str(age.N_h) ...                                                      %number of units in the recurrent net
        ' --filedata ' age.path2navigation dataname '.mat' ...                              %file with the data
        ' --sparsity ' num2str(age.sparsity) ...                                            %weight of sparsity contraint
        ' --filesave ' filesave]);                                                          %file where to save the data 

system(['python netnp.py --epochs ' num2str(N_epochs) ...                                     %number of epochs to train
        ' --epoch_ini ' num2str(age.last_epoch) ...                                         %initial epoch
        ' --N_h ' num2str(age.N_h) ...                                                      %number of units in the recurrent net
        ' --filedata ' age.path2navigation dataname '.mat' ...                              %file with the data
        ' --sparsity ' num2str(age.sparsity) ...                                            %weight of sparsity contraint
        ' --filesave ' filesavenp]);                                                          %file where to save the data 

    
age.last_epoch = age.last_epoch + N_epochs;

% end

%% Load data of the final epoch and update internal states

%loads the weights saved during the training in the agent
epoch = 800;
% fileweights = [filesave 'weights_Ep' num2str(epoch) '.mat'];
% fileweights = [age.path2train 'openmaze64_sparsity0weights_Ep400.mat']; 
fileweights = [age.path2train 'openmaze64_sparsity0_seed1_orthW_pred_Nh100predweights_Ep110.mat'];
% fileweights = [age.path2train 'openmaze64_sparsity0_seed1_nopred_Nh500weights_Ep716.mat'];
age.weights = load(fileweights);

%recompute the internal states of the network for the random exploration
age = age.updatehq();   


%% Plot environment
figure('Name', 'World');
imagesc(env.world);
age.save_figure('World',[simulation_name '_World']);

%% Plot example trajectory
%trajectory for 70 steps of the agent
age.fig_trajectory(age.hist.s,1:numel(age.hist.s), 70, '_example');
age.save_figure('trajectory_example',[simulation_name '_Trajetory_example']);

%% Plot place and direction fields for 100 neurons
timepoints = 1000000;
age = age.select(1:timepoints);
timepoints_eval = 50000;      %time points per epoch to calculate hidden activity
rng(0)
sample = datasample(1:timepoints, timepoints_eval, 'replace', false);
[del idx] = sort(mean(age.hist.h(sample,:)),'descend'); %this gets the 100 most active neurons
age.fig_placecells(idx(1:100), sample, ['_Ep' num2str(epoch)]);
% age.save_figure(['placecells_Ep' num2str(epoch)],[simulation_name '_PlaceCells_Ep' num2str(epoch)]);
age.fig_directioncells(idx(1:100), sample, ['_Ep' num2str(epoch)]);
% age.save_figure(['directioncells_Ep' num2str(epoch)],[simulation_name '_DirCells_Ep' num2str(epoch)]);

%% Plots in PCA of the latent variables x,y,theta
% get the position in xy coordinate from the state states.
age_temp = age;
age_temp.select(sample);
variables = mat2cell(age_temp.s2p(age_temp.hist.s),numel(age_temp.hist.s),[1 1]);
% get the action
variables{3} = age_temp.hist.d;

% plot the first 3 pca components in 3D with the color for x,y,a4
pcas = [1,2,3];
age_temp.fig_pca(pcas,variables,sample,'123xya');
% age_temp.save_figure('pca123xya',[simulation_name '_PCA123xya']);

% plot the pca 4 and 5 in 2D with the color for x,y,a
pcas = [4,5,6];
age_temp.fig_pca(pcas,variables,sample,'45xya');
view(2)
% age_temp.save_figure('pca45xya',[simulation_name '_PCA45xya']);

%% Plots in PCA representation colored by observations
% get the position in xy coordinate from the state states.
% Mdist = mean(age_temp.hist.o(:,1:5),2);
% MRcol = mean(age_temp.hist.o(:,6:10),2);
% Mgcol = mean(age_temp.hist.o(:,11:15),2);
% MBcol = mean(age_temp.hist.o(:,16:20),2);

Mdist = pca(age_temp.hist.o(:,1:5)'); Mdist = Mdist(:,1);
MRcol = pca(age_temp.hist.o(:,6:10)'); MRcol = MRcol(:,1);
Mgcol = pca(age_temp.hist.o(:,11:15)'); Mgcol = Mgcol(:,1);
MBcol = pca(age_temp.hist.o(:,16:20)'); MBcol = MBcol(:,1);
% [del ObsPC] = pca(age_temp.hist.o);

variables = mat2cell([Mdist, MRcol, Mgcol, MBcol], size(age_temp.hist.o,1),[1 1 1 1]);
% variables = mat2cell([ObsPC(:,1),ObsPC(:,2)], size(age_temp.hist.o,1),[1 1]);

% plot the first 3 pca components in 3D with the color for x,y,a4
pcas = [1,2,3];
age_temp.fig_pca(pcas,variables,sample,'123xya');
% age_temp.save_figure('pca123xya',[simulation_name '_PCA123xyaobs']);

% plot the pca 4 and 5 in 2D with the color for x,y,a
pcas = [4,5,6];
age_temp.fig_pca(pcas,variables,sample,'45xya');
% age_temp.save_figure('pca45xya',[simulation_name '_PCA45xyaobs']);

%% Plot in PCA of place fields
% axis_pcao = pca(age_temp.hist.o(:,6:20));
% pcao = (axis_pcao(:,1:2)' * age_temp.hist.o')';o

pcas = [1,2];
idxs = idx(1:100);%[2,12,15,22]);
activity_neurons = mat2cell(age_temp.hist.h(:,idxs), numel(age_temp.hist.s), ones(1, numel(idxs)));
% Mdist = mean(age_temp.hist.o(:,1:5),2);
age_temp.fig_pca(pcas,activity_neurons,sample,'12placecellsobs');


%% Plot in PCA of observations
pcas = [1,2];
idxs = idx(1:100);%[2,12,15,22]);
activity_neurons = mat2cell(age_temp.hist.h(:,idxs), numel(age_temp.hist.s), ones(1, numel(idxs)));
age_temp.fig_pca(pcas, activity_neurons,sample,'2placecells');
% age_temp.save_figure('pca2placecells',[simulation_name '_pca2placecells']);

%% Define Entropy stuff
% [d, n] = size(X);
sigmaX = @(X) sqrt(trace(cov(X))/size(X,1));    
silverman = @(X) sigmaX(X) * (4/((size(X,1)+2)*size(X,2)))^(1/(size(X,1)+4)); %d = dim, n = N_points
logmean =@(X) -log(mean(X(:)));

stdmat = @(x) arrayfun(@(i_row) std(x(i_row,:)), 1:size(x,1));
vertvec = @(x) x(:);
normalize2idm = @(x) 1/size(x,1) * x./(sqrt(diag(x)*diag(x)')+eps);
Halpha = @(alpha, X) 1/(1-alpha) * log2(sum(real(eigs(X,20,'lm','Tolerance',1e-10,'IsSymmetricDefinit',1)).^alpha));
Hjoint = @(alpha, X, Y) Halpha(alpha, X.*Y/trace(X.*Y));

C1 = (2*pi)^(-1);
C2 = @(X,sigma) -1/(sigma*sigmaX(X) * sqrt(2));                  
Kxy = @(X,Y) slmetric_pw(X, Y, 'sqdist');
Gij = @(X,Y,sigma) C1 * exp(C2(X,sigma) * Kxy(X,Y));


%% Compute quantities over learning loading weights for each epoch
Cov = @(x) (x'*x)/size(x,1);
tr2C = @(Cov) sqrt(trace(Cov^2)/size(Cov,1));
dimC = @(Cov) (trace(Cov)^2) / trace((Cov)^2);
dimCov = @(X) dimC(cov(X'));
normalization = 0;
PackNum = @(X) intrinsic_dim(X', 'PackingNumbers', normalization);
GMST = @(X) intrinsic_dim(X', 'GMST', normalization);
% MLE = @(X) intrinsic_dim(X', 'MLE', normalization);
EV = @(X) intrinsic_dim(X', 'EigValue', normalization);
CorrDim = @(X) intrinsic_dim(X', 'CorrDim');
NearNb = @(X) intrinsic_dim(X', 'NearNbDim', normalization);

% estimators = {'dimCov', 'MiND_ML','MLE','DANCoFit'}%, 'EV', 'CorrDim'};%,'MiND_KL','PackNum'};%, 'GMST','MiND_ML',,'MiND_KL', 'DANCo'
estimators = {'dimCov', 'MiND_ML','MLE', 'DANCoFit', 'CorrDim', 'GMST'};%,'MiND_KL'};%, 'MiND_ML',,'MiND_KL', 'DANCo'

idEst = [];
spentTime = [];

epochs_set = 1:10:111;     %set of epochs to consider
timepoints = 100000;
timepoints_eval = 1000;      %time points per epoch to calculate hidden activity
rng(0)
sample = datasample(1:timepoints, timepoints_eval, 'replace', false);
age_full = age;         %store the agent before reducing the timepoints in its history
age = age.select(1:timepoints);
dims = zeros(numel(epochs_set),numel(estimators));
sigma = .4;

input = zscore(age.hist.o(sample,:))';
variables = mat2cell(age.s2p(age.hist.s(sample)),numel(age.hist.s(sample)),[1 1]);
latent = zscore([variables{:}, age.hist.d(sample)])';
GijII = Gij(input,input, sigma);
H2I = logmean(GijII);
H2Ii = Halpha(2,normalize2idm(GijII));
GijLL = Gij(latent,latent, sigma);
H2L = logmean(GijLL);
H2Li = Halpha(2,normalize2idm(GijLL));

%%
for i_epoch = 1:numel(epochs_set)
    disp(['This is epoch ' num2str(epochs_set(i_epoch))])
    
    %load weights and calculate activity for the epoch
%     fileweights = [filesave 'weights_Ep' num2str(epochs_set(i_epoch)) '.mat'];
    fileweights = [age.path2train 'openmaze64_sparsity0weights_Ep' num2str(epochs_set(i_epoch)) '.mat'];
%     fileweights = [age.path2train 'openmaze64_sparsity0_seed1_nopred_Nh500weights_Ep' num2str(epochs_set(i_epoch)) '.mat'];
    age.weights = load(fileweights);
    age_temp = age;    
    age_temp.select(1:timepoints);    
    age_temp = age_temp.updatehq();                    
    
    %compute all quantities of interest (PCA, CCA, Costs, Eigenvectors...)
    hidden = age_temp.hist.h';
    hidden = hidden(:,sample);
    [axis, del, scores(i_epoch,:)] = pca(hidden');        
    pca_var90(i_epoch) = sum(cumsum(scores(i_epoch,:))/sum(scores(i_epoch,:))<.90);
    pca_vartot(i_epoch) = sum(abs(scores(i_epoch,:)));    
    pca_activity = (axis(:,1:5)' * hidden)';
    [A,B,rxy1(i_epoch,:)] = canoncorr(pca_activity(:,1:3),age_temp.s2p(age_temp.hist.s(sample)));    
    [A,B,rtheta1(i_epoch,:)] = canoncorr(pca_activity(:,1:3),age_temp.hist.d(sample));
    [A,B,rxy2(i_epoch,:)] = canoncorr(pca_activity(:,4:5),age_temp.s2p(age_temp.hist.s(sample)));    
    [A,B,rtheta2(i_epoch,:)] = canoncorr(pca_activity(:,4:5),age_temp.hist.d(sample));   
    
    axis_pcao = pca(age_temp.hist.o);
    pcao = (axis_pcao(:,1:2)' * age_temp.hist.o(sample,:)')';

    [A,B,robs1(i_epoch,:)] = canoncorr(pca_activity(:,1:3), pcao);       
    [A,B,robs2(i_epoch,:)] = canoncorr(pca_activity(:,4:5), pcao);           
    
    eigens(i_epoch,:) = eig(age_temp.weights.rec_weight);
    cost(i_epoch) = mean(age_temp.hist.cost_prediction);
    cost_s(i_epoch) = mean(age_temp.hist.cost_sparsity);
    cost_sparsity(i_epoch) = mean(age_temp.hist.cost_sparsity);
    
%     GijXX = Gij(hidden,hidden,sigma);
%     H2X(i_epoch) = logmean(GijXX);
%     H2Xi(i_epoch) = Halpha(2,normalize2idm(GijXX));       
%     
%     H2XIi(i_epoch) = Hjoint(2,normalize2idm(GijXX),normalize2idm(GijII));
%     MXIi(i_epoch) = H2Xi(i_epoch)+H2Ii-H2XIi(i_epoch);
%     
%     H2XLi(i_epoch) = Hjoint(2,normalize2idm(GijXX),normalize2idm(GijLL));
%     MXLi(i_epoch) = H2Xi(i_epoch)+H2Li-H2XLi(i_epoch);    
    
    
    for i_est = 1:numel(estimators)
        %estimator = str2func(estimators{i_est}); idEst = estimator(ratessel);
        tic;        
%         hiddent = age_temp.hist.o';
        try idEst(i_est) = eval([estimators{i_est}, '(hidden)']);  catch idEst(i_est)=nan; end
        try idEstW(i_est) = eval([estimators{i_est}, '(age_temp.weights.rec_weight)']);  catch idEst(i_est)=nan; end
        spentTime(i_est) = toc;
        dims(i_epoch, i_est) = idEst(i_est);
%         dimsW(i_epoch, i_est) = idEstW(i_est);
        %                 if any(isnan(idEst)) || any(idEst>N_space(i_N)), pause; end
    end
        
end



%% Cost function & Trace of correlation matrix (Variability measure)
figure('Name','Cost_Corr');

subplot(221)
hold on
% Cost function
semilogy(epochs_set, cost(1:numel(epochs_set)),'linewidth',2);
semilogy(epochs_set, cost_sparsity(1:numel(epochs_set)),'linewidth',2);
yyaxis right

% Trace of the correlation matrix
plot(epochs_set, pca_var90(1:numel(epochs_set)), 'linewidth',2)
legend('Cost','Tot Var')
hold off
%%
% figure()
subplot(222)
cla
hold on
plot(epochs_set, H2X(1:numel(epochs_set)), 'linewidth',2)
plot(epochs_set, H2Xi(1:numel(epochs_set)), 'linewidth',2)
plot(epochs_set, MXIi(1:numel(epochs_set)), 'linewidth',2)
plot(epochs_set, MXLi(1:numel(epochs_set)), 'linewidth',2)
hold off
%%
clf
subplot(321)
cla
hold on
plot(epochs_set, dims(1:numel(epochs_set),1),'linewidth',2);
% plot(epochs_set, pca_var90(1:numel(epochs_set)), 'linewidth',2)
% plot(epochs_set, 10*mean(abs(r(1:numel(epochs_set),:)')), 'linewidth',2)
hold off
legend('PR');%, '# PC 90%')
ylim([0, 20]);


subplot(322)
cla
hold on
plot(epochs_set, dims(1:numel(epochs_set),2),'linewidth',2);
plot(epochs_set, dims(1:numel(epochs_set),3),'linewidth',2);
plot(epochs_set, dims(1:numel(epochs_set),4),'linewidth',2);
plot(epochs_set, dims(1:numel(epochs_set),5),'linewidth',2);
plot(epochs_set, dims(1:numel(epochs_set),6),'linewidth',2);
hold off
ylim([0, 20]);
legend(estimators{2:end})

subplot(323)
cla
hold on
plot(epochs_set, dimsW(1:numel(epochs_set),1),'linewidth',2);
hold off
legend('PR');%, '# PC 90%')

subplot(324)
cla
hold on
% plot(epochs_set, dimsW(1:numel(epochs_set),1),'linewidth',2);
plot(epochs_set, dimsW(1:numel(epochs_set),2),'linewidth',2);
plot(epochs_set, dimsW(1:numel(epochs_set),3),'linewidth',2);
plot(epochs_set, dimsW(1:numel(epochs_set),4),'linewidth',2);
plot(epochs_set, dimsW(1:numel(epochs_set),5),'linewidth',2);
plot(epochs_set, dimsW(1:numel(epochs_set),6),'linewidth',2);
hold off
legend(estimators{2:end})
ylim([0 500])
% age_temp.save_figure('Cost_Corr',[simulation_name '_Cost_Corr']);


subplot(325)
cla
hold on
plot(epochs_set, dimsW(1:numel(epochs_set),1)./mean(dimsW(1:numel(epochs_set),[2,3,5]),2),'linewidth',2);
hold off
legend('PR');%, '# PC 90%')
% ylim([0,100]);

subplot(326)
cla
hold on
plot(epochs_set, dims(1:numel(epochs_set),1)./mean(dims(1:numel(epochs_set),[2,3,4,5]),2),'linewidth',2);
hold off
legend(estimators{2:end})
% ylim([0,100]);
% age_temp.save_figure('Cost_Corr',[simulation_name '_Cost_Corr']);


%% Canonical covariance and number of PCAs for 90% variability
figure('Name','PCA90_CCA');
subplot(221)
hold on
% plot(epochs_set, pca_var90(1:numel(epochs_set)),'linewidth',2);
% set(gca,'yticklabel',cellfun(@(x) sprintf('%5.1e',exp(str2num(x))), get(gca,'YTicklabel'),'uniformoutput',false))
plot(epochs_set, mean(abs(rxy1(1:numel(epochs_set),:)')), 'linewidth',2)
plot(epochs_set, (abs(rtheta1(1:numel(epochs_set),:)')), 'linewidth',2)
plot(epochs_set, mean(abs(rxy2(1:numel(epochs_set),:)')), 'linewidth',2)
plot(epochs_set, (abs(rtheta2(1:numel(epochs_set),:)')), 'linewidth',2)
% plot(epochs_set, mean(abs(robs1(1:numel(epochs_set),:)')), 'linewidth',2)
% plot(epochs_set, mean(abs(robs2(1:numel(epochs_set),:)')), 'linewidth',2)
% legend('PCA 90%','CCA')
legend('XY on PC 1,2,3','XY on PC 4,5','\theta on PC 1,2,3','\theta on PC 4,5');
% errorbar(1:N_epochs,1./sigma,1./sigma-1./(sigma_ustd),1./(sigma_lstd)-1./(sigma), 'linewidth',2)
hold off
% age.save_figure('PCA90_CCA',[simulation_name '_PCA90_CCA']);

subplot(222)
hold on
plot(epochs_set, mean(abs(robs1(1:numel(epochs_set),:)')), 'linewidth',2)
plot(epochs_set, mean(abs(robs2(1:numel(epochs_set),:)')), 'linewidth',2)
legend('PCA 90%','CCA')
legend('Obs on PC 1,2,3','Obs PC 4,5');
% errorbar(1:N_epochs,1./sigma,1./sigma-1./(sigma_ustd),1./(sigma_lstd)-1./(sigma), 'linewidth',2)
hold off

% age.save_figure('PCA90_CCA',[simulation_name '_PCA90_CCA']);



%% RL to learn to navigate towards the rewards
age = age_full;
age.gamma = 0.5;            % discount rate of RL
age.T = 10;                % intial temperature (noise). The temperature decreases over trials with discount 0.99
age.alpha = 0.005;          % initial learning rate. The learning rate decreases over trials with discount 0.995
age_reset = age;            

% set the reward in a random place
rng(1);
env.rewards = zeros(age.N_x,age.N_y);    
pos_ini_randx = rand(1);
pos_ini_randy = rand(1);
env.rewards(round((pos_ini_randx*9)/10*age.N_x):round((pos_ini_randx*9)/10*age.N_x)+5, round((pos_ini_randy*9)/10*age.N_y):round((pos_ini_randy*9)/10*age.N_y)+5) = age.r;         
env.rewards = env.rewards(:);

%% Q learning for the agent. At each trial the agent is set in a random place and needs to reach the goal    
trials_per_set = 100;       %the total number of trials is split into set to monitor the learning
number_of_sets = 15;
max_steps = 1000;           %maximum number fo step for the agent to arrive to the reward.
age_Qlearn = age;           %this agent will learn with Q-learning
age_Lookup = age;           %this agent will learn with a Lookup table

for i_set_of_trials = 1:15            
    age_Qlearn = age_Qlearn.explore_Qlearn(env,trials_per_set, max_steps, false, false);
    % Uncommeting the line below plots the trajectories of the agent during learning.
    % age = age.explore_reward(env,trials_per_set, max_steps,true, true);
    age_Lookup = age_Lookup.explore_Lookup(env,trials_per_set, max_steps, false, false);
    
    % Monitor number of steps to reward
    figure(1);
    clf; hold on;
    plot(movavg(age_Qlearn.hist.steps,10,10));
    plot(movavg(age_Lookup.hist.steps,10,10));
    hold off;
    
    % Monitor values of theta for Q learning
    figure(2); clf;
    subplot(121);
    imagesc(age_Qlearn.theta_Qlearn_hist);colorbar;drawnow();    
    subplot(122);
    imagesc(age_Lookup.theta_Lookup_hist);colorbar;drawnow();    
    
    % Monitor ValueMap buildup
    age_Qlearn = age_Qlearn.updatehq();
    age.fig_mapvalue(age_Qlearn.hist.s, age_Qlearn.hist.Q, 100000,'_Qlearning');    
    age.fig_mapvalue(age_Lookup.hist.s, age_Lookup.hist.Q, 100000,'_Llearning');    
    
    %Monitor most important place cells
    [del idx] = sort(age_Qlearn.theta_Qlearn,'descend');
    fig = age_Qlearn.fig_placecells(idx(1:16), 100000,'_Qlearning');
    
    drawnow();    
end

%%
age.save_figure('mapvalue_Qlearning',[simulation_name '_Qvaluemap']);
age.save_figure('placecells_Qlearning',[simulation_name '_Qplacecells']);
age.save_figure('steps_Qlearning',[simulation_name '_Qsteps']);





















%% Plotting place cells and direction cells per epoch during learning
dir_content = dir;
filenames = {dir_content.name};
current_files = filenames;
while true
  dir_content = dir;
  filenames = {dir_content.name};
  new_files = setdiff(filenames,current_files);
  if ~isempty(new_files)    
    epoch = regexp(new_files{end},'\d+\.?\d*|-\d+\.?\d*|\.?\d*','match');
    epoch = round(str2num(epoch{end}));
    current_files = filenames;
    
    fprintf(['plots of epoch' num2str(epoch) '\n'])
    datanameweights = 'data641rmscalllr300';
    age = age.load_weights(datanameweights, epoch);
    age = age.load2agent(dataname,false);
    age = age.updatehq();
    
    age.fig_weights();
    age.fig_netactivity(1000);
    age.fig_trajectory(age.hist.s,100);
    idx=1:144;
    age.fig_placecells(idx,1000000,epoch);
    age.fig_directioncells(idx,1000000, epoch);
    pause(120)
  else
    fprintf('no new files\n')
    pause(60)    
  end
end


%% Averaging over predictive or non-predictive networks
timepoints = 100000;
sample = datasample(1:timepoints, 1000);
age = age.select(1:timepoints);
dir_content = dir;
filenames = {dir_content.name};
current_files = filenames;
allWfeed = [];
allWrec = [];
allWrecfw = [];
eigsWrec = [];
dimW = [];

%%
for i_net = 1:100          
    if i_net<51
        filename = ['openmaze64_sparsity0_seed' num2str(i_net) '_pred_Nh100predweights'];
    else
        filename = ['openmaze64_sparsity0_seed' num2str(i_net-50) '_pred_Nh100nopredweights'];
    end
    netfiles = filenames(contains(filenames,filename));
    if ~isempty(netfiles)        
    allepochs = regexp(netfiles,'Ep\d*','match');
    maxepoch = max(cellfun(@(x) str2num(x{1}(3:end)),allepochs,'uni',1));      
    maxeps(i_net) = maxepoch;

%     age = age.load2agent(dataname,false); 
    if maxepoch < 100
        epset = 1:maxepoch
    else
        epset = 1:100;%round(linspace(1,maxepoch,100));
    end
    for i_ep = 1:numel(epset)            
        [i_net,i_ep]
        age = age.updatehq();
        datanameweights = [filename num2str(epset(i_ep)) '.mat'];
        age = age.load_weights(filename, epset(i_ep));        
        allWfeed(i_net, i_ep, :,:) = zscore(age.weights.out_weight * age.weights.inp_weight(:,1:20));
        allWrecfw(i_net, i_ep, :,:) = zscore(age.weights.out_weight * age.weights.rec_weight);
        allWrec(i_net, i_ep, :,:) = zscore(age.weights.rec_weight);
        eigsWrec(i_net, i_ep,:) = abs(eig(age.weights.rec_weight));
        dimW(i_net, i_ep) = sum(eigsWrec(i_net, i_ep,:))^2/sum(eigsWrec(i_net, i_ep,:).^2);
        hidden = age.hist.h(sample,:);%end-N_steps+1:end
        axis_pca = pca(hidden);
        axis_pca = axis_pca(:,1:5);
        pca_activity = (axis_pca' * hidden')';                        
        [A,B,rxy1(i_net,i_ep,:)] = canoncorr(pca_activity(:,1:3),age.s2p(age.hist.s(sample)));    
        [A,B,rtheta1(i_net,i_ep,:)] = canoncorr(pca_activity(:,1:3),age.hist.d(sample));
        [A,B,rxy2(i_net,i_ep,:)] = canoncorr(pca_activity(:,4:5),age.s2p(age.hist.s(sample)));    
        [A,B,rtheta2(i_net,i_ep,:)] = canoncorr(pca_activity(:,4:5),age.hist.d(sample));   
        
        axis_pcao = pca(age.hist.o);
        pcao = (axis_pcao(:,1:2)' * age.hist.o(sample,:)')';
        [A,B,robs1(i_net,i_ep,:)] = canoncorr(pca_activity(:,1:3), pcao);       
        [A,B,robs2(i_net,i_ep,:)] = canoncorr(pca_activity(:,4:5), pcao);  
        
        dimPR(i_net,i_ep) = dimC(cov(hidden));
%         dimGMST(i_net,i_ep) = GMST(hidden');     
        dimDanco(i_net,i_ep) = DANCoFit(hidden');
        dimCorr(i_net,i_ep) = CorrDim(hidden');        
        dimMLE(i_net,i_ep) = MLE(hidden');        
    end
%     age.fig_weights();
%     age.fig_netactivity(1000);
%     age.fig_trajectory(age.hist.s,100);
%     idx=1:144;
%     age.fig_placecells(idx,1000000,epoch);
%     age.fig_directioncells(idx,1000000, epoch);
%     imagesc(squeeze(allWfeed(i_net,:,:)));colormap('jet');colorbar
%     pause()
    else      
    continue;
  end
end
%%

imagesc(squeeze(mean(allWfeed,1)));colormap('jet');colorbar
imagesc(squeeze(mean(allWrecfw,1)));colormap('jet');colorbar
abs(eig(squeeze(allWrec(1,:,:))))
brighten(0.5)

%%
predcol = [0.12, 0.47, 0.70];
nonpredcol =  [0.89, 0.10, 0.11];
cols=colormap('lines');
col1p = [6, 174, 213]/256;
col2p = [143, 201, 58]/256;
col3p = [204, 41, 54]/256;
col4p = [0, 60, 113]/256;
col5p = [153, 221, 200]/256;
col1n=[250, 244, 228]/256;
col2n=[254, 213, 187]/256;
col3n=[241, 157, 123]/256;
col4n=[145, 132, 130]/256;
col5n=[135, 198, 189]/256;

avp = 1;
avn = .5;
ava= .4;
figure(1)
clf()
subplot(231)
hold on
pl1=plot(mean(rxy1(1:50,2:1:end,:),3)','color',cols(5,:),'linewidth',1);
for i_x=1:numel(pl1) eval(['pl1(' num2str(i_x) ').Color(4)=avp;']); end
plm1 = plot(nanmean(nanmean(rxy1(1:50,2:1:end,:),1),3)','--','color','black','linewidth',2); plm1.Color(4)=ava;
pl2=plot(mean(robs1(1:50,2:1:end,:),3)','color',cols(6,:),'linewidth',1);
for i_x=1:numel(pl2) eval(['pl2(' num2str(i_x) ').Color(4)=avp;']); end
plm2 = plot(nanmean(nanmean(robs1(1:50,2:1:end,:),1),3)','--','color','black','linewidth',2); plm2.Color(4)=ava;
ylim([0,1]);

subplot(232)
hold on
for i_x=1:200 eval(['rxy1(i_x,' num2str(maxeps(i_x)) ':end,:)=NaN;']); end
for i_x=1:200 eval(['robs1(i_x,' num2str(maxeps(i_x)) ':end,:)=NaN;']); end
pl1=plot(mean(rxy1([51:74,76:100],2:end,:),3)','color',cols(5,:),'linewidth',1);
for i_x=1:numel(pl1) eval(['pl1(' num2str(i_x) ').Color(4)=avn;']); end
plm1 = plot(nanmean(nanmean(rxy1([51:74,76:100],2:1:end,:),1),3)','--','color','black','linewidth',2); plm1.Color(4)=ava;
pl2=plot(nanmean(robs1([51:74,76:100],2:end,:),3)','color',cols(6,:),'linewidth',1);
for i_x=1:numel(pl2) eval(['pl2(' num2str(i_x) ').Color(4)=avn;']); end
plm2 = plot(nanmean(nanmean(robs1([51:74,76:100],2:end,:),1),3)','--','color','black','linewidth',2); plm2.Color(4)=ava;
ylim([0,1]);
xlim([0,80]);

subplot(234)
hold on
pl1=plot(dimPR(1:50,2:end)','color',cols(3,:),'linewidth',1);
for i_x=1:numel(pl1) eval(['pl1(' num2str(i_x) ').Color(4)=avp;']); end
plm1 = plot((mean(dimPR(1:50,2:end),1))','--','color','black','linewidth',2); plm1.Color(4)=ava;
dimNL = (dimMLE(1:50,2:end)'+dimCorr(1:50,2:end)'+dimDanco(1:50,2:end)')/3
pl2=plot(dimNL,'color',cols(4,:),'linewidth',1);
for i_x=1:numel(pl2) eval(['pl2(' num2str(i_x) ').Color(4)=avp;']); end
plm2 = plot((mean(dimNL,2))','--','color','black','linewidth',2); plm2.Color(4)=ava;
ylim([0,16]);

subplot(235)
hold on
for i_x=1:200 eval(['dimPR(i_x,' num2str(maxeps(i_x)) ':end,:)=NaN;']); end
dimNL = (dimMLE+dimCorr+dimDanco)/3;
for i_x=1:200 eval(['dimNL(i_x,' num2str(maxeps(i_x)) ':end,:)=NaN;']); end
pl1=plot(dimPR([51:74,76:100],2:end)','color',cols(3,:),'linewidth',1);
for i_x=1:numel(pl1) eval(['pl1(' num2str(i_x) ').Color(4)=avn;']); end
plm1 = plot((nanmean(dimPR([51:74,76:100],2:end),1))','--','color','black','linewidth',2); plm1.Color(4)=ava;
pl2=plot(dimNL([51:74,76:100],2:end)','color',cols(4,:),'linewidth',1);
for i_x=1:numel(pl2) eval(['pl2(' num2str(i_x) ').Color(4)=avn;']); end
plm2 = plot((nanmean(dimNL([51:74,76:100],2:end)',2)),'--','color','black','linewidth',2); plm2.Color(4)=ava;
ylim([0,16]);
xlim([0,80]);

subplot(236)
cla
hold on
DG = dimPR(:,:)./dimNL(:,:);
pl1=plot(DG(1:50,2:end)','color',predcol,'linewidth',1);
for i_x=1:numel(pl1) eval(['pl1(' num2str(i_x) ').Color(4)=avp;']); end
plm1 = plot((nanmean(DG(1:50,2:end),1))','--','color','black','linewidth',2); plm1.Color(4)=ava;
pl2=plot(DG([51:74,76:100],2:end)','color',nonpredcol,'linewidth',1);
for i_x=1:numel(pl2) eval(['pl2(' num2str(i_x) ').Color(4)=avn;']); end
plm2 = plot(nanmean(DG([51:74,76:100],2:end)),'--','color','black','linewidth',2); plm2.Color(4)=ava;
ylim([0,4]);






