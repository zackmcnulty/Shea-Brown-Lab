classdef environment
    
% environment is an objet with a structure containing fields:
% actions: the possible actions that can be accomplished on the environment
% places: the possible places of the environment
% observations: the list of observations in all the different places
% rewards: the rewards in the environment
% step: is a function that takes as argument an observation and an
% action generating the next position;

    
    properties        
        actions;
        places;
        rewards;
        observations;                        
        N_x;
        N_y;
        N_a;
        range;
        walls;
        world;
        stoch_move;
        colcorr;
        sensors_max;
        sensors_min;
        reward_quantity; 
    end
    
    methods
        % methods, including the constructor are defined in this block
        
        function obj = environment(walls, correlation)
            % class constructor
            if(nargin > 0)
                obj.N_x = size(walls,1);
                obj.N_y = size(walls,2);                
                obj.places = 1:obj.N_x * obj.N_y;                
                obj.walls = walls;
            end            
            obj.range = ceil(sqrt(obj.N_x^2 + obj.N_y^2));
            obj.N_a = 8;
            obj.actions = [0, 0; 1,0; 1,1; 0,1; -1,1; -1,0; -1,-1; 0,-1; 1,-1];             
            obj.observations = [];
            obj.stoch_move = 0;
            obj.colcorr = correlation;                     
            obj.sensors_max = sqrt((obj.N_x-1)^2+(obj.N_y-1)^2);
            obj.sensors_min = 1;
            obj.world = obj.coloring(obj.walls);
            obj.reward_quantity = 10;
            obj.rewards = zeros(obj.N_x,obj.N_y);
            obj.rewards(round(1/3*obj.N_x):round(1/3*obj.N_x)+5, round(1/3*obj.N_y):round(1/3*obj.N_y)+5) = obj.reward_quantity; 
            obj.rewards = obj.rewards(:);              
        end
        
        function [new_position, new_action_idx, new_observation, new_reward] = step(obj, position, direction, angle_sensors)   
            action = d2a(direction);                                       
            new_position = position + obj.actions(action,:);
            new_position = max(new_position,[0 0]);
            new_position = min(new_position,[obj.N_x, obj.N_y]);
            if any([new_position<=0, new_position>=[obj.N_x, obj.N_y]]) == 1;
                new_position = position;            
            else if obj.walls(new_position(1),new_position(2)) == 1;
                new_position = position; 
                end
            end        
            if rand(1) < obj.stoch_move,            
                new_position = position;                
            end
            new_action = new_position - position;
            [del new_action_idx] = ismember(new_action,obj.actions,'rows');
            
            [sensors_dist, sensors_hit] = obj.sensors_range(new_position, direction, obj.walls>0, angle_sensors);
            sensors_hits = sub2ind([obj.N_x, obj.N_y],sensors_hit(:,1),sensors_hit(:,2));                                    
%             new_observation = [sensors_dist, sensors_hits];
            cols = hits2cols(sensors_hits,obj.world);
            new_observation = [sensors_dist, cols];
            new_observation = new_observation(:);                        
            new_reward = obj.rewards(sub2ind([obj.N_x, obj.N_y],round(new_position(1)),round(new_position(2))));                                               
        end                       
        
        function [sensors_dist, sensors_hit] = sensors_range(obj, position, angle, walls, angle_sensors)
            num_sensors = numel(angle_sensors);
            sensors_dist = zeros(num_sensors,1);
            sensors_hit = zeros(num_sensors,2);         
            for i_sen = 1:num_sensors                                
                for i_range = 0.5:1:obj.range
                    ray = zeros(1,2);
                    ray = round([position(1) position(2)] + i_range * [cos(angle + angle_sensors(i_sen)) sin(angle + angle_sensors(i_sen))]);
                    if any([ray<=0, ray>size(walls)]), break
                    elseif any([walls(ray(1), ray(2))])
                        sensors_hit(i_sen,:) = [ray(1), ray(2)];%/sum(World(ray(1), ray(2),:));%/255
                        sensors_dist(i_sen,1) = i_range;
                        break
                    end
                end
            end
            sensors_dist = ((sensors_dist-obj.sensors_min)/(obj.sensors_max-obj.sensors_min));                                   
        end
                
        function world = coloring(obj, walls)            
            rng(1);
            Colors_rand = double(rand(obj.N_x + 2*obj.colcorr,obj.N_y + 2*obj.colcorr,3)>0.5);
            h = fspecial('disk',obj.colcorr);
            Colors=imfilter(Colors_rand, h);
            Colors=Colors(obj.colcorr:obj.colcorr+obj.N_x-1, obj.colcorr:(obj.colcorr+obj.N_y-1),:);                        
            world = tanh(((Colors-mean(Colors(:)))/std(Colors(:)))+1)/2;                  
            world(:,:,1) = world(:,:,1) .* walls;
            world(:,:,2) = world(:,:,2) .* walls;
            world(:,:,3) = world(:,:,3) .* walls;            
        end
        
        
%         function [sensors_dist, sensors_hit] = sensors_range_new(obj, position, angle, walls, angle_sensors)
%             num_sensors = numel(angle_sensors);            
%             B = bwboundaries(walls);         
%             sensors_dist = zeros(num_sensors,1);
%             sensors_hit = zeros(num_sensors,2);
%             for i_sen = 1:num_sensors                
%                 intersections = [];
%                 ray = round([position(1), position(2)] + obj.range * [cos(angle + angle_sensors(i_sen)) sin(angle + angle_sensors(i_sen))]);
%                 for i_B = 1:numel(B) 
%                     [xi, yi] = polyxpoly([position(1), ray(1)], [position(2), ray(2)], B{i_B}(:,1), B{i_B}(:,2));
%                     intersections = [intersections; [xi,yi]];
%                 end
%                 distances = sqrt(sum((intersections-[position(1), position(2)]).^2,2));
%                 [del idx] = min(distances);
%                 sensors_hit(i_sen,:) = round([intersections(idx,1), intersections(idx,2)]);
%                 sensors_dist(i_sen,1) = distances(idx);
%             end        
%             sensors_min = 1;
%             sensors_max = 82;
%             sensors_dist = ((sensors_dist-sensors_min)/(sensors_max-sensors_min));
%         end
                
    end       
end

function new_action = d2a(direction)
            [del new_action]=min(abs(direction - 2 * pi/8 *(0:8))); 
            if new_action == 9, new_action=1; end
            new_action = new_action + 1;
end

function cols = hits2cols(idxs, walls)
    walls_reshaped = reshape(walls, size(walls,1)*size(walls,2),3);
    cols = walls_reshaped(idxs,:);                
end


