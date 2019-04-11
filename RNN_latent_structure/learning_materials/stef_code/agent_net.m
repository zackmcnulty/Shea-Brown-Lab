classdef agent_net

    % This class represents the agent. It has several methods:
    % - explore_randomly
    
    % - p2s(p)  converts position p to state s
    % - s2p(s)  converts state s to position s
    % - hidden(h,o,a) computes the next step of hidden activity given the last
    %       hidden activity h, observation o, action a.
    % - hidden1(h,o,a) computes the hidden activity in 2 steps given the last
    %       hidden activity h, observation o, action a. It uses the output
    %       of the network and feeds it back in it.
    % - [hidden, output] = net(h_ini, obsevations, actions)  computes the
    %       activity and output of the network for a stream of observations and
    %       actions.
    % - explore_randomly(env, N_steps, record) makes the agent explore the 
    %       environment env for N_steps. If record is true the observations
    %       and actions are stored in the structure hist of the agent.
    % - update()  updates all the values of the history structure hist in
    %       the agent. Given the stream of observations and actions it
    %       recalculates the values of the hidden activity and Q values for
    %       the weights of the network. 
    % - explore_Qlearn(env, N_trials, max_steps, record, plot) 
    %       Learns a Q-learning RL function from the hidden activity of the
    %       network to the actions. The learning proceeds for N_trials
    %       trials and each trial can have at least max_steps number of
    %       steps of the agent. Record and plot are flags to record or plot
    %       the exploration of the agent during the learning.
    % - explore_Lookup(env, N_trials, max_steps, record, plot) 
    %       Same as Qlearn but for a lookup table representation.
    % - load2agent(dataname, append) loads the data in the agent history.
    %       If append is true then the data are appended to the present
    %       history.
    % - save2train(N_steps, dataname) saves the last N_steps steps of the
    %       agent into a file that can be loaded through load2agent.
    % - load_weights(dataname, epoch) it loads a weight file generated 
    %       through the learning into the weights structure of the agent.
    %       The new hidden states can then be computed via the updatehq
    %       function.
    
    % Several methods are for plotting:
    % - fig_trajectory  plots the trajectory of the agent in the
    %       environment.
    % - fig_mapvalue  plots a value map of a given quantity on the
    %       environment.
    % - fig_netactivity(N_steps)
    % - fig_weights()
    % - fig_pca(pcas, variables, N_steps, varargin)
    % - fig_placecells(idx, N_steps, varargin)
    % - fig_directioncells(idx, N_steps, varargin)
    
    
    properties        
        N_x;
        N_y;
        p;
        p_ini;
        a;        
        s;
        d;
        o;
        h;        
        r;  
        Q;
        N_s; %number of states
        N_a; %number of actions 
        N_trials;
        a_onehot;
        N_o; 
        N_h;               
        N_sensors;
        theta_Qlearn;
        theta_Lookup
        alpha;
        T;
        eps_greedy;
        gamma;
        steps;
        weights;                
        Q_values;
        theta_Qlearn_hist;
        theta_Lookup_hist;
        angle_sensors;
        s_ini;
        a_ini;               
        actions;        
        stoch_angle;
        reg;
        transfer;
        figs;
        alpha_orthogonal
        hist;
        last_save;
        last_epoch;
        path2navigation;
        path2train;    
        path2figures;    
        sparsity;
    end
    
    methods
        % methods, including the constructor are defined in this block              
        
        function obj = agent_net(size_env)
            % class constructor             
            obj.N_s = size_env(1)*size_env(2);                
            obj.N_x = size_env(1);
            obj.N_y = size_env(2);
            obj.p_ini = [round(obj.N_x/3),round(obj.N_y/3)];
            obj.s_ini = obj.p2s(obj.p_ini);
            obj.a_ini = 4;                        
            obj.s = obj.s_ini;
            obj.a = obj.a_ini;           
            obj.N_sensors = 40;
            obj.N_a = 9;                                
            obj.r = 100;       
            obj.Q = 0;            
            obj.N_h = 500;
            obj.N_o = obj.N_sensors*4;
            obj.h = zeros(1,obj.N_h);  
            obj.o = zeros(1,obj.N_o);  
            obj.weights = obj.initialize_weights();                       
            obj.N_trials = 100;
            obj.a_onehot = eye(obj.N_a);
            obj.theta_Qlearn = zeros(1,obj.N_h);
            obj.theta_Lookup = zeros(obj.N_s,obj.N_a);
            obj.steps = 0;            
            obj.alpha = 0.01;
            obj.eps_greedy = 1;
            obj.T = 100;
            obj.gamma = 0.6;
            obj.stoch_angle = 0.5;
            obj.reg = 1e-6;
            obj.alpha_orthogonal = 1;
            obj.gamma = 0.6;                          
            obj.Q_values = zeros(obj.N_s,obj.N_a);
            obj.theta_Qlearn_hist = [];
            obj.theta_Lookup_hist = [];
            obj.angle_sensors = linspace(-pi/4,pi/4,obj.N_sensors);
            obj.d = a2d(obj.a_ini);
            obj.transfer = @(x) 1./(1+exp(-x));
            obj.actions = [0, 0; 1,0; 1,1; 0,1; -1,1; -1,0; -1,-1; 0,-1; 1,-1];   
            obj.hist = [];         
            obj.hist.steps = [];
            obj.hist.a = obj.a;
            obj.hist.s = obj.s;
            obj.hist.d = obj.d;
            obj.hist.o = obj.o;
            obj.hist.h = obj.h;
            obj.hist.Q = obj.Q;
            obj.last_save = [];
            obj.last_epoch = 1;
            obj.path2navigation = 'data/navigation/';
            obj.path2train = 'data/train/';            
            obj.path2figures = 'data/figures/';            
            obj.sparsity = 0;
        end                        
        
        function s = p2s(obj,p)
            s = sub2ind([obj.N_x, obj.N_y], p(:,1),p(:,2));
        end
        
        function pos = s2p(obj, s)
            [pos(:,1), pos(:,2)] = ind2sub([obj.N_x, obj.N_y], s);
        end
                        
        function hid = hidden(obj,h,o,a)
            hid = cell2mat(arrayfun(@(x) obj.transfer(obj.weights.rec_weight * h' + obj.weights.rec_bias + obj.weights.inp_weight * [o,obj.a_onehot(x,:)]'), a,'uniformoutput',false));    
        end
        
        function hid_1 = hidden_1(obj, h,o,a)
            hid_1 = cell2mat(arrayfun(@(x) obj.transfer(obj.weights.rec_weight * h' + obj.weights.rec_bias + obj.weights.inp_weight * [(obj.weights.out_weight*obj.transfer(obj.weights.rec_weight * h' + obj.weights.rec_bias + obj.weights.inp_weight * [o, obj.a_onehot(x,:)]'))', obj.a_onehot(x,:)]'), a,'uniformoutput',false));    
        end
        
        function [hidden, output] = net(obj, h_ini, observations, actions)            
            W_rec = obj.weights.rec_weight;
            W_inp = obj.weights.inp_weight;
            W_out = obj.weights.out_weight;
            b_rec = obj.weights.rec_bias;
            b_out = obj.weights.out_bias;
            a_onehot = obj.a_onehot;
            hidden = zeros(numel(actions),obj.N_h);
            hidden(1,:) = h_ini;
            for i_t = 2:numel(actions)
                hidden(i_t,:) = obj.transfer(W_rec * hidden(i_t-1,:)' + b_rec + W_inp * [observations(i_t,:), a_onehot(actions(i_t),:)]')';                        
            end
            output = (W_out * hidden' + b_out)';
        end
        
         function obj = set_Nh(obj, N_h)            
             obj.N_h = N_h;
             obj.theta_Qlearn = zeros(1,obj.N_h);
             obj.h = zeros(1,obj.N_h);                   
             obj.hist.h = obj.h;
%              obj.weights = obj.initialize_weights();
        end             
        
        
        function obj = explore_randomly(obj, env, N_steps, record)            
            a_trial = zeros(N_steps,1);
            s_trial = zeros(N_steps,1);
            d_trial = zeros(N_steps,1);
%             h_trial = zeros(N_steps,obj.N_h);
            o_trial = zeros(N_steps,obj.N_o);
            Q_trial = zeros(N_steps,1);
            for i_steps = 1:N_steps
                
                % Get new state after taking action (s_1,r_1, a) 
                pos = obj.s2p(obj.s);
                if obj.a == 1, obj.d = obj.d+1*rand(1);end
                [p_1, obj.a, o_1, r_1] = env.step([pos(1), pos(2)], obj.d, obj.angle_sensors); % the action is reassigned as might be corrected by environment boundaries
                s_1 = obj.p2s(p_1);
                det_a0 = obj.a==1;
                
                % Calculate which of the possible actions has highest value
                as = possible_actions(obj.d);
                Q_values = rand(size(as)); %random policy
                idx_chosen = choose(Q_values,obj.T);
                a_chosen = as(idx_chosen);
                obj.a = a_chosen;
                
                % Take the step by updating internal variables
                obj.s = s_1;
                d_1 = new_d(mod(obj.d + obj.stoch_angle*det_a0*(2*randi([0,1])-1),2*pi), obj.a, obj.stoch_angle  *(1+det_a0));
                obj.d = d_1;
                obj.a = d2a(mod(obj.d,2*pi))*~det_a0+det_a0;                
                obj.o = o_1';
                
                if record
                    a_trial(i_steps,1) = obj.a;
                    s_trial(i_steps,1) = obj.s;                    
                    d_trial(i_steps,1) = obj.d;                                        
                    o_trial(i_steps,:) = obj.o;                          
                end                
            end
%             [h_trial, out_trial] = obj.net(obj.hist.h(end,:), o_trial, a_trial);                    
%             Q_trial = (obj.theta_Qlearn * h_trial')';                                                                        
            obj.hist.a = [obj.hist.a; a_trial];
            obj.hist.s = [obj.hist.s; s_trial];
            obj.hist.d = [obj.hist.d; d_trial];            
            obj.hist.o = [obj.hist.o; o_trial];            
%             obj.hist.h = [obj.hist.h; h_trial];            
%             obj.hist.Q = [obj.hist.Q; Q_trial];            
        end     
        
        function obj = explore_Qlearn(obj, env, N_trials, max_steps, record, plot)                                    
            theta = obj.theta_Qlearn;
            for i_trial = 1:N_trials
                theta_trial = zeros(size(obj.theta_Qlearn));
                obj.s = datasample(find(~((env.rewards+env.walls(:))>0)==1),1);
                a_trial = zeros(max_steps,1);
                s_trial = zeros(max_steps,1);
                d_trial = zeros(max_steps,1);
                o_trial = zeros(max_steps,obj.N_o);
                Q_trial = zeros(max_steps,1);
                h_trial = zeros(max_steps,obj.N_h);
                for i_steps = 1:max_steps
                    
                    % Get new state after taking action (s_1,r_1, a)
                    pos = obj.s2p(obj.s);
                    if obj.a == 1, obj.d = obj.d+1*rand(1);end
                    [p_1, obj.a, o_1, r_1] = env.step([pos(1), pos(2)], obj.d, obj.angle_sensors); % the action is reassigned as might be corrected by environment boundaries
                    s_1 = obj.p2s(p_1);
                    h_1 = obj.hidden(obj.h, obj.o, obj.a)';
                    det_a0 = obj.a==1;
                    
                    % Calculate which of the possible actions has highest value
                    as = obj.possible_actions(obj.d);
                    Q_values = theta*obj.hidden_1(h_1, o_1', as); %representation induced policy
                    Q_1max = max(Q_values);
                    a_maxs = find(Q_values == Q_1max);                    
                    a_1max = as(datasample(a_maxs,1));
                    
                    idx_chosen = obj.choose(Q_values,obj.T);
                    a_chosen = as(idx_chosen);
                    Q_chosen = Q_values(idx_chosen);
                    if isempty(a_chosen), a_chosen = a_1max;end
                    
                    Q_value = theta * obj.hidden(obj.h,obj.o,obj.a);
                    obj.a = a_chosen;
                    
%                     Q_mean = mean(Q_values);
                    delta = r_1 + obj.gamma * Q_1max - Q_value;
%                     delta = r_1 + obj.gamma * Q_mean - Q_value;
                    if and(delta,~det_a0) 
                        theta_trial = theta_trial + obj.alpha * delta *obj.h;
                    end
                    
                    
                    % Take the step by updating internal variables
                    obj.s = s_1;
                    obj.r = r_1;
                    d_1 = obj.new_d(mod(obj.d + obj.stoch_angle*det_a0*(2*randi([0,1])-1),2*pi), obj.a, obj.stoch_angle  *(1+det_a0));
                    obj.d = d_1;
                    obj.a = obj.d2a(mod(obj.d,2*pi))*~det_a0+det_a0;
                    obj.o = o_1';
                    obj.h = h_1;
                    
                    if record                                 
                        a_trial(i_steps) = obj.a;
                        s_trial(i_steps) = obj.s;
                        d_trial(i_steps) = obj.d;
                        o_trial(i_steps,:) = obj.o;
                        Q_trial(i_steps) = (theta * obj.h');
                        h_trial(i_steps,:) = obj.h;
                    end
                    if obj.r ~= 0                     
                        a_trial(i_steps+1:end) = [];
                        s_trial(i_steps+1:end) = [];
                        d_trial(i_steps+1:end) = [];
                        o_trial(i_steps+1:end,:) = [];
                        Q_trial(i_steps+1:end) = [];
                        h_trial(i_steps+1:end,:) = [];                        
                        theta = theta + theta_trial;
                        break; 
                    end;
                end
        
                
                if plot                             
                    fig_trajectory(obj,s_trial,1:numel(s_trial), i_steps-1, 'time');
                    drawnow()                    
                    fig_trajectory(obj,s_trial, Q_trial, i_steps-1, 'value');
                    drawnow()
%                    pause()                
                end                                                                   
                                
                if record                    
                    obj.hist.a = [obj.hist.a; a_trial];
                    obj.hist.s = [obj.hist.s; s_trial];
                    obj.hist.d = [obj.hist.d; d_trial];
                    obj.hist.o = [obj.hist.o; o_trial];
                    obj.hist.h = [obj.hist.h; h_trial];
                    obj.hist.Q = [obj.hist.Q; Q_trial];
                end
                
%                 theta = theta/(sum(theta.^2));
                obj.theta_Qlearn_hist = [obj.theta_Qlearn_hist, theta'];
                
                % age.eps_greedy = age.eps_greedy * 0.99;
                 obj.T = obj.T * 0.99 +0.001;
                 obj.alpha = obj.alpha * 0.995 + 0.000000025;                
%                 steps = numel(s_trial(diff(s_trial)~=0))+1;
                obj.hist.steps = [obj.hist.steps; i_steps];                
                disp(['Trial number ' num2str(i_trial) ' steps ' num2str(i_steps) ' T ' num2str(obj.T)])
            end             
            obj.theta_Qlearn = theta;
        end
        

        function obj = explore_Lookup(obj, env, N_trials, max_steps, record, plot)
            walls = sum(env.walls,3);
            theta = obj.theta_Lookup;
            for i_trial = 1:N_trials
                theta_trial = zeros(size(obj.theta_Lookup));                
                obj.s = datasample(find(~((env.rewards+walls(:))>0)==1),1);
                a_trial = zeros(max_steps,1);
                s_trial = zeros(max_steps,1);
                d_trial = zeros(max_steps,1);
                o_trial = zeros(max_steps,obj.N_o);
                Q_trial = zeros(max_steps,1);
                for i_steps = 1:max_steps
                    Q_query = theta_trial;
                    
                    % Get new state after taking action (s_1,r_1, a)
                    pos = obj.s2p(obj.s);
                    if obj.a == 1, obj.d = obj.d+1*rand(1);end
                    [p_1, obj.a, o_1, r_1] = env.step([pos(1), pos(2)], obj.d, obj.angle_sensors); % the action is reassigned as might be corrected by environment boundaries
                    s_1 = obj.p2s(p_1);                  
                    det_a0 = obj.a==1;

                    
                    % Calculate which of the possible actions has highest value
                    as = obj.possible_actions(obj.d);
                    Q_1max = max(Q_query(s_1,as));
                    a_maxs = find(Q_query(s_1,as) == Q_1max);
                    a_1max = as(datasample(a_maxs,1));
                    
                    % Update value of theta with RL signal
                    Q_value = Q_query(obj.s,obj.a);
                    delta = r_1 + obj.gamma * Q_1max - Q_value;
                    if delta, theta_trial(obj.s, obj.a) = theta_trial(obj.s, obj.a) + obj.alpha * delta; end
                    
                    % Take the step by updating internal variables
                    obj.s = s_1;
                    obj.r = r_1;
                    if rand(1) > 1-obj.eps_greedy
                        obj.a = a_1max; 
                    else
                        obj.a = as(randi(numel(as))); 
                    end
                    
                    % Take the step by updating internal variables
                    obj.s = s_1;
                    obj.r = r_1;
                    d_1 = obj.new_d(mod(obj.d + obj.stoch_angle*det_a0*(2*randi([0,1])-1),2*pi), obj.a, obj.stoch_angle  *(1+det_a0));
                    obj.d = d_1;
                    obj.a = obj.d2a(mod(obj.d,2*pi))*~det_a0+det_a0;
                    obj.o = o_1';                    
                    
                    if record                                 
                        a_trial(i_steps) = obj.a;
                        s_trial(i_steps) = obj.s;
                        d_trial(i_steps) = obj.d;
                        o_trial(i_steps,:) = obj.o;
                        Q_trial(i_steps) = Q_value;                        
                    end
                    if obj.r ~= 0                     
                        a_trial(i_steps+1:end) = [];
                        s_trial(i_steps+1:end) = [];
                        d_trial(i_steps+1:end) = [];
                        o_trial(i_steps+1:end,:) = [];
                        Q_trial(i_steps+1:end) = [];
                        theta = theta + theta_trial;
                        break; 
                    end;
                end
        
                
                if plot                             
                    fig_trajectory(obj,s_trial,1:numel(s_trial), i_steps-1, 'time');
                    drawnow()                    
                    fig_trajectory(obj,s_trial, Q_trial, i_steps-1, 'value');
                    drawnow()
%                    pause()                
                end                                                                   
                                
                if record                    
                    obj.hist.a = [obj.hist.a; a_trial];
                    obj.hist.s = [obj.hist.s; s_trial];
                    obj.hist.d = [obj.hist.d; d_trial];
                    obj.hist.o = [obj.hist.o; o_trial];                    
                    obj.hist.Q = [obj.hist.Q; Q_trial];
                end
                
                obj.theta_Lookup_hist = [obj.theta_Lookup_hist, theta(:)'];
                                
                obj.hist.steps = [obj.hist.steps; i_steps];                
                disp(['Trial number ' num2str(i_trial) ' steps ' num2str(i_steps) ' T ' num2str(obj.T)])
            end             
            obj.theta_Lookup = theta;
        end
                    
        
        function obj = save2train(obj, N_steps, dataname)
           data.observations = obj.hist.o(end-N_steps:end,:);
           data.actions = obj.hist.a(end-N_steps:end)-1;
           data.states = obj.hist.s(end-N_steps:end);
           data.directions = obj.hist.d(end-N_steps:end);
           data.weights = obj.weights;           
           save([obj.path2navigation dataname],'-v6','-struct','data');
           obj.last_save = dataname;
        end                        
              
        function obj = load2agent(obj, dataname, append)
           data = load([obj.path2navigation dataname]);
           if append
                obj.hist.o = [obj.hist.o; data.observations];
                obj.hist.a = [obj.hist.a; data.actions+1];
                obj.hist.d = [obj.hist.d; data.directions];
                obj.hist.s = [obj.hist.s; data.states];
           else               
               obj.hist.o = data.observations;
               obj.hist.a = data.actions+1;
               obj.hist.d = data.directions;
               obj.hist.s = data.states;
           end
        end
        
        function obj = updatehq(obj)           
           [h_hist, out_hist] = obj.net(obj.hist.h(end,:), obj.hist.o, obj.hist.a);                    
           Q_hist = (obj.theta_Qlearn * h_hist')';                                                                        
           obj.hist.h = h_hist; 
           obj.hist.Q = Q_hist;   
           obj.hist.out = out_hist;   
           obj.hist.cost_prediction = mean((obj.hist.out(1:end-1,:) - obj.hist.o(2:end,:)).^2,2);
           obj.hist.cost_prediction(end+1,:) = obj.hist.cost_prediction(end,:)*0;
           obj.hist.cost_sparsity = mean(abs(obj.hist.h),2);
        end   
        
        
        function obj = select(obj, timepoints)           
            obj.hist.a = obj.hist.a(timepoints);
            obj.hist.s = obj.hist.s(timepoints);
            obj.hist.d = obj.hist.d(timepoints);
            obj.hist.o = obj.hist.o(timepoints,:);                        
            if size(obj.hist.h,1) > numel(timepoints)
                obj.hist.h = obj.hist.h(timepoints,:);
            end
            if numel(obj.hist.Q) > numel(timepoints)
                obj.hist.Q = obj.hist.Q(timepoints);  
            end            
        end   
        
        function obj = train(obj, dataname, epoch_ini, N_epochs)                                         
            system(['python net.py --epochs ' num2str(N_epochs) ' --epoch_ini ' num2str(epoch_ini) ' --N_h ' num2str(obj.N_h) ' --filename ' dataname '.mat']);                      
            obj.last_epoch = obj.last_epoch+N_epochs-1;
        end                        
        
        function obj = load_weights(obj, dataname, epoch)
            obj.weights = load([obj.path2train dataname '_Ep' num2str(epoch) '.mat']);
        end
        
        function weights = initialize_weights(obj)
            weights.out_bias = single(zeros(obj.N_o,1));
            weights.rec_bias = single(zeros(obj.N_h,1));
            weights.inp_weight = single(0.02*randn(obj.N_h, obj.N_o+obj.N_a));
            weights.out_weight = single(0.02*randn(obj.N_o, obj.N_h));            
            weights.rec_weight = single(eye(obj.N_h));
        end
                                                       
        function fig = fig_trajectory(obj, s_hist, colors, N_steps, varargin)
            if nargin > 3
                fig_tag = varargin{1}; 
            else 
                fig_tag = '';
            end
            name= ['trajectory' fig_tag];                       
            fig = sfigure(fig_name(name));
            clf;
            pos = obj.s2p(s_hist(end-N_steps:end)');
            cols = colors(end-N_steps:end);
            scatter(pos(:,1),pos(:,2),20,cols);
            xlim([0,obj.N_x+1]);
            ylim([0,obj.N_y+1]);
            colormap('jet');
            drawnow();
        end          
        
        function fig = fig_mapvalue(obj, s_hist, q_hist, N_steps, varargin)
             if nargin > 3
                tag = varargin{1}; 
            else 
                tag = [];
            end
            name= ['mapvalue' num2str(tag)];                                               
            fig = sfigure(fig_name(name));
            clf;
            positions = obj.s2p(s_hist(end-N_steps+1:end)');            
            Q_values = q_hist(end-N_steps+1:end,:);
            x_min = min(positions(positions(:,1)~=0,1));
            x_max = max(positions(:,1));
            y_min = min(positions(positions(:,2)~=0,1));
            y_max = max(positions(:,2));
            x_space = double(x_min:1:x_max);
            y_space = double(y_min:1.0:y_max);
            x_idx = arrayfun(@(x) find((x_space-x)>=0,1), double(positions(:,1)));
            y_idx = arrayfun(@(y) find((y_space-y)>=0,1), double(positions(:,2)));                        
            place_valueavg = NaN * zeros(obj.N_x, obj.N_y);
            place_valuestd = NaN * zeros(obj.N_x, obj.N_y);           
            for x_pos = 1:obj.N_x
                for y_pos = 1:obj.N_y
                    idx_pos = find((x_idx==x_pos).*(y_idx==y_pos));
                    if ~isempty(idx_pos)  
                        place_valueavg(x_pos, y_pos) = mean(Q_values(idx_pos));
                        place_valuestd(x_pos, y_pos) = std(Q_values(idx_pos));
                    end
                end
            end                        
            imagesc((squeeze(place_valueavg)'));
            set(gca,'YDir','normal');
            colorbar;
        end                
        
        function fig = fig_netactivity(obj, N_steps)
            name= 'netactivity';                        
            fig = sfigure(fig_name(name));
            clf;
            subplot(4,1,1)
            imagesc(obj.hist.h(end-N_steps:end,:)');colorbar
            subplot(4,1,2)
            imagesc(obj.hist.o(end-N_steps:end,:)');colorbar;
            subplot(4,1,3)
            imagesc(obj.hist.o(end-N_steps:end,:)' - obj.hist.out(end-N_steps:end,:)');colorbar;            
            subplot(4,1,4)
            imagesc(obj.hist.a(end-N_steps:end,:)');colorbar;
            colormap('jet')            
        end
        
        function fig = fig_weights(obj)            
            name= 'weights';                                    
            fig = sfigure(fig_name(name));
            clf;
            subplot(2,3,1)
            imagesc(obj.weights.rec_weight-eye(size(obj.weights.rec_weight)));colorbar
            subplot(2,3,2)
            imagesc(obj.weights.inp_weight);colorbar;
            subplot(2,3,3)
            imagesc(obj.weights.out_weight);colorbar
            subplot(2,3,4)
            imagesc(obj.weights.rec_bias);colorbar
            subplot(2,3,5)
            imagesc(obj.weights.out_bias);colorbar
            colormap('jet')
        end
        
        function fig = fig_pca(obj, pcas, variables, sample, varargin) %N_steps
            if nargin > 3
                fig_tag = varargin{1}; 
            else 
                fig_tag = [];
            end
            name= ['pca' fig_tag];     
            sfigure(fig_name(name));
            fig=panel(fig_name(name));
            clf
            num_cols = max([3, ceil(sqrt(numel(variables)))]);
            num_rows = ceil(numel(variables)/num_cols);
            fig.pack(num_rows, num_cols);
            fig.de.margin=1;            
            hidden = obj.hist.h(sample,:);%end-N_steps+1:end
            axis_pca = pca(hidden);
            axis_pca = axis_pca(:,pcas);
            pca_activity = (axis_pca' * hidden')';
            
            N_bins = round(sqrt(4096));
            [N,Xedges,Yedges,binX,binY] = histcounts2(pca_activity(:,1),pca_activity(:,2),N_bins);            
            posX = Xedges(1) + cumsum(diff(Xedges)/2);
            posY = Yedges(1) + cumsum(diff(Yedges)/2);      
            idxthreshhist = 1:sum(N(:)>0);%find(N(N(:)>0)>0);            
            tabvar=table(binX, binY);        
            tabvar = tabvar(idxthreshhist,:);
            [G binXidx binYidx] = findgroups(tabvar.binX,tabvar.binY);            
                        
            H = fspecial('gaussian',10,3);
            for i_var = 1:numel(variables)
                plotvar = NaN * N; 
                variable = variables{i_var}(sample);%end-N_steps+1:end               
                [sub_1, sub_2] = ind2sub([num_rows,num_cols],i_var);
                fig(sub_1, sub_2).select();
                if numel(pcas)==2                                
                    meanvar=splitapply(@(x) mean(x),variable(idxthreshhist),G);                    
                    plotvar = full(sparse(binXidx,binYidx,meanvar, size(N,1), size(N,2)));                    
                    im = imfilter(plotvar,H,'replicate');
                    im(N==0) = NaN;
%                     im(im==0) = NaN;
                    set(gca,'xticklabel',[]);
                    set(gca,'yticklabel',[]);                
                    axis('off');
                    ylim([+0.5,size(N,1)+0.5]);
                    xlim([+0.5,size(N,2)+0.5]); 
                    pcolor(im);                    
                    set(gca, 'ydir', 'reverse');
                    shading flat                    
%                     colorbar
%                     scatter(posX(binXidx),posY(binYidx),25,meanvar,'filled','s');
                    caxis(prctile(im(:),[5,95]));
                    caxis([min(im(:)),max(im(:))]);                   
%                     xlim([min(posX),max(posX)]);
%                     ylim([min(posY),max(posY)]);
%                     scatter(pca_activity(idxvar,1),pca_activity(idxvar,2),.02,variable(idxvar),'.');                    
                else
                    [del idxvar] = sort(variable, 'ascend');
                    scatter3(pca_activity(idxvar,1),pca_activity(idxvar,2),pca_activity(idxvar,3),.02,variable(idxvar),'.');
                    zlim([min(pca_activity(idxvar,3)),max(pca_activity(idxvar,3))]);   
                    xlim([min(pca_activity(:,1)),max(pca_activity(:,1))]);
                    ylim([min(pca_activity(:,2)),max(pca_activity(:,2))]);
                    view([0,2.5,6]);         
                    grid('off');
                    axis('off');                                    
                end
            end
            colormap(jet);
        end
        
        function fig = fig_placecells(obj, idx, sample, varargin) %N_steps
            if nargin > 2
                tag = varargin{1}; 
            else 
                tag = [];
            end
            name= ['placecells' tag];     
            sfigure(fig_name(name));
            fig=panel(fig_name(name));
            clf
            fig.pack(ceil(sqrt(numel(idx))),ceil(numel(idx)/ceil(sqrt(numel(idx)))));
            fig.de.margin=1;            
            positions = obj.s2p(obj.hist.s(sample));%end-N_steps+1:end));
            hidden = obj.hist.h(sample,:);%end-N_steps+1:end,:);
            x_min = min(positions(positions(:,1)~=0,1));
            x_max = max(positions(:,1));
            y_min = min(positions(positions(:,2)~=0,1));
            y_max = max(positions(:,2));
            x_space = double(x_min:1:x_max);
            y_space = double(y_min:1.0:y_max);
            x_idx = arrayfun(@(x) find((x_space-x)>=0,1), double(positions(:,1)));
            y_idx = arrayfun(@(y) find((y_space-y)>=0,1), double(positions(:,2)));            
            place_activity = NaN*ones(x_max,y_max,numel(idx));
            for x_pos = x_min:x_max
                for y_pos = y_min:y_max
                    idx_pos = find((x_idx==x_pos).*(y_idx==y_pos));                                        
                    place_activity(x_pos, y_pos,1:numel(idx)) = mean(hidden(idx_pos,idx),1);
                end
            end                        
            for i_neu = 1:numel(idx)
                [sub_1, sub_2] = ind2sub([ceil(sqrt(numel(idx))),ceil(numel(idx)/ceil(sqrt(numel(idx))))],i_neu);
                fig(sub_1, sub_2).select();
                caxis([min(min(place_activity(:,:,i_neu))),max(max(place_activity(:,:,i_neu)))]);
%                 caxis([0,max(max(place_activity(:)))/2]);
                set(gca,'xticklabel',[]);
                set(gca,'yticklabel',[]);                
                axis('off');
                ylim([+0.5,obj.N_y+0.5]);
                xlim([+0.5,obj.N_x+0.5]);                                                
%                 caxis([0,.1])
                imagesc((squeeze(place_activity(:,:,i_neu))'));
            end
            drawnow
            colormap('jet');
        end
        
        
        function fig = fig_directioncells(obj, idx, N_steps, varargin)
            % Directions Cells
            if nargin > 2
                fig_offset = varargin{1}; 
            else 
                fig_offset = [];
            end
            name= ['directioncells' num2str(fig_offset)];     
            sfigure(fig_name(name));
            fig=panel(fig_name(name));
            clf
            row_num = ceil(sqrt(numel(idx)));
            col_num = ceil(numel(idx)/ceil(sqrt(numel(idx))));
            fig.de.margin=1;            
        
            N_sectors = 60;
            a_space = linspace(0,2*pi, N_sectors);
            angles = obj.hist.d(end-N_steps+1:end);
            hidden = obj.hist.h(end-N_steps+1:end,:);

            angle_activity = NaN*ones(N_sectors, numel(idx));
            for i_ang = 1:N_sectors-1
                idx_ang = find((angles>=a_space(i_ang)).*(angles<=a_space(i_ang+1)));
                angle_activity(i_ang,:) = mean(hidden(idx_ang,idx),1);
            end
            
%             angle_activity_norm = (angle_activity - min(angle_activity(:)))/(max(angle_activity(:)-min(angle_activity(:))));            
            
            for i_neu = 1:numel(idx)                
                pax = subplot(row_num, col_num, i_neu, polaraxes);
                angle_activity_norm_neu = (angle_activity(:,i_neu) - min(angle_activity(:,i_neu)))/(max(angle_activity(:,i_neu)-min(angle_activity(:,i_neu))));
                polarplot(pax,a_space, angle_activity_norm_neu,'LineWidth',2);
                set(gca,'RTick',[]);
                set(gca,'ThetaTick',[]);                
                rlim([0,1])
%               This below is the old 'patchy' solution. 
%                 for i_ang = 1:N_sectors-1
%                     alpha = a_space(i_ang):0.01:a_space(i_ang+1)+0.01;
%                     patch([0 cos(alpha) 0], [0 sin(alpha) 0], angle_activity_norm(i_ang,i_neu),'edgecolor','none');
%                 end
            end
        end                        
             
        
        %% Copy of functions for debugging
        function as = possible_actions(obj,d)
            as = mod(d2a(d)-2 + [-2,-1,0,1,2],8) + 2;
        end
        
        function idx = choose(obj, x,T)
            idx = find(((cumsum(exp(x/T)))/sum(exp(x/T))-rand(1))>0,1,'first');
        end
        
        function d_1 = new_d(obj, d,a, stoch_angle)
            d_a = a2d(a,d);
            sign_diff = sign(mod(d_a-d + pi,2*pi)-pi);
            d_diff = sign_diff * stoch_angle * abs(randn(1));
            d_1 = mod(d + d_diff,2*pi);
        end
        
        function d = a2d(obj, a,d_last)
            directions = linspace(0,7/8*2*pi,8);
            if a == 1
                d = d_last;
            else
                d = directions(round(a-1));
            end
        end
        
        function a = d2a(obj, d)
            [del a]=min(abs(d - 2 * pi/8 *(0:8)));
            if a == 9, a = 1; end
            a = a+1;
        end
        
        function del = save_figure(obj, namefig, namesave)
            handle = findobj(allchild(groot), 'flat', 'type', 'figure', 'Name', namefig);
            saveas(handle, [obj.path2figures namesave],'epsc');        
        end               
        
    end
end

function handle = fig_name(name)
            handle = findobj(allchild(groot), 'flat', 'type', 'figure', 'Name', name);
            if isempty(handle)
                handle = sfigure('Name',name);
            end             
        end

function d_1 = new_d(d,a, stoch_angle)
    d_a = a2d(a,d);
    sign_diff = sign(mod(d_a-d + pi,2*pi)-pi);
    d_diff = sign_diff * stoch_angle * abs(randn(1));
    d_1 = mod(d + d_diff,2*pi);
end


function idx = choose(x,T)
idx = find(((cumsum(exp(x/T)))/sum(exp(x/T))-rand(1))>0,1,'first');
end

function d = a2d(a,d_last)
    directions = linspace(0,7/8*2*pi,8);
    if a == 1         
        d = d_last;
    else
        d = directions(round(a-1));
    end
end

function a = d2a(d)
    [del a]=min(abs(d - 2 * pi/8 *(0:8)));
    if a == 9, a = 1; end
    a = a+1;
end

function as = possible_actions(d)
    as = mod(d2a(d)-2 + [-1,0,1],8) + 2;
end

function h = sfigure(varargin)
if nargin>=1
    h = varargin{1};
    if ishandle(h)
        set(0, 'CurrentFigure', h);
    else
        par = '';
        for i=1:nargin
            var = sprintf ('var%d',i);
            par = sprintf ('%s,%s',par,var);
            eval (sprintf('%s = varargin{%d};',var,i));
        end
        par = par (2:length(par));
        eval (sprintf('h = figure(%s);',par));
    end
else
    par = '';
    for i=1:nargin
        var = sprintf ('var%d',i);
        par = sprintf ('%s,%s',par,var);
        eval (sprintf('%s = varargin{%d};',var,i));
    end
    par = par (2:length(par));
    eval (sprintf('h = figure(%s);',par));
end
end

