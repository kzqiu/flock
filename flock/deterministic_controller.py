import numpy as np

class DeterministicController:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        
    def get_actions(self, observation, env):
        """Generate actions using a deterministic policy for cooperative transport"""
        # extract global information about object and target
        obs_length = len(observation)
        global_info_start = obs_length - 6
        
        # object position and velocity are normalized by width/height
        obj_pos_x = observation[global_info_start] * env.width
        obj_pos_y = observation[global_info_start + 1] * env.height
        target_pos_x = observation[global_info_start + 4] * env.width
        target_pos_y = observation[global_info_start + 5] * env.height
        
        # calculate direction from object to target
        obj_pos = np.array([obj_pos_x, obj_pos_y])
        target_pos = np.array([target_pos_x, target_pos_y])
        obj_to_target = target_pos - obj_pos
        dist_to_target = np.linalg.norm(obj_to_target)
        
        if dist_to_target > 0.001:
            obj_to_target_norm = obj_to_target / dist_to_target
        else:
            obj_to_target_norm = np.array([1.0, 0.0])  # default direction if at target
            
        # direction perpendicular to object-target line for pushing from sides
        perp_dir = np.array([-obj_to_target_norm[1], obj_to_target_norm[0]])
        
        # generate actions for each agent
        actions = np.zeros((self.num_agents, 2))
        agent_obs_dim = 11
        
        # process each agent
        for i in range(self.num_agents):
            start_idx = i * agent_obs_dim
            agent_pos_x = observation[start_idx] * env.width
            agent_pos_y = observation[start_idx + 1] * env.height
            agent_pos = np.array([agent_pos_x, agent_pos_y])
            
            # vector from agent to object and distance
            agent_to_obj = obj_pos - agent_pos
            dist_to_obj = np.linalg.norm(agent_to_obj)
            
            if dist_to_obj > 0.001:
                agent_to_obj_norm = agent_to_obj / dist_to_obj
            else:
                agent_to_obj_norm = np.array([0.0, 0.0])
            
            # get obstacle information
            obstacle_dist = observation[start_idx + 8]
            obstacle_dir_x = observation[start_idx + 9]
            obstacle_dir_y = observation[start_idx + 10]
            obstacle_dir = np.array([obstacle_dir_x, obstacle_dir_y])
            
            action = agent_to_obj_norm
            
            # avoid obstacles if theyre very close
            if obstacle_dist < 0.3:
                action = obstacle_dir * 1.0
            
            # simple positioning based on agent index
            elif dist_to_obj > env.transport_object.width * 2.0:
                offset_angle = (i / self.num_agents) * np.pi * 2
                offset_dir = np.array([np.cos(offset_angle), np.sin(offset_angle)]) * 0.2
                action = agent_to_obj_norm * 0.8 + offset_dir * 0.2
            
            # if close to object, determine role based on position
            else:
                rel_pos = agent_pos - obj_pos
                angle_to_target = np.arctan2(obj_to_target_norm[1], obj_to_target_norm[0])
                agent_angle = np.arctan2(rel_pos[1], rel_pos[0])
                rel_angle = (agent_angle - angle_to_target + np.pi) % (2 * np.pi) - np.pi
                
                # agent is behind the object
                if abs(rel_angle) > np.pi * 0.6:
                    action = obj_to_target_norm
                    
                # agent is in front of the object
                elif abs(rel_angle) < np.pi * 0.3:
                    # move to either side and back
                    side = 1 if i % 2 == 0 else -1
                    action = -obj_to_target_norm * 0.6 + perp_dir * side * 0.8
                    
                # agent is on the sides
                else:
                    # apply diagonal force
                    side_factor = np.sign(rel_angle)
                    action = obj_to_target_norm * 0.7 - perp_dir * side_factor * 0.7
            
            actions[i] = self._normalize_action(action)
                
        return actions
        
    def _normalize_action(self, action):
        """Normalize the action vector"""
        norm = np.linalg.norm(action)
        if norm > 0.001:
            return action / norm
        return action