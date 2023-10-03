function rules = get_rules(fis, features, targets)
    
    % Initialize the rule matrix that contains 4 memership functions index for all inputs,
    % the weight and membership function index for the output 
    rules_matrix = zeros(size(features, 1), size(features, 2) + 2);

    % Iterate all samples in feature matrix
    for i = 1 : size(features, 1)
    
        weight = 1;
    
        % Iterate all features of a sample
        for j = 1 : size(features, 2)
    
            membership_function_index = 0;
            membership_degree = 0;
    
            % Compute membership degree for every membership function
            for k = 1 : size(fis.Inputs(j).MembershipFunctions, 2)
                
                membership_function_parameters = fis.Inputs(j).MembershipFunctions(k).Parameters;
                new_membership_degree = trimf(features(i, j), membership_function_parameters);
    
                % Search for the highest membership degree value
                if (new_membership_degree > membership_degree)
                    membership_degree = new_membership_degree;
                    membership_function_index = k;
                end
            end
           
            % Update the weight of the rule and save the chosen membership degree value
            weight = weight * membership_degree;
            rules_matrix(i, j) = membership_function_index;
        end
    
        % Add weight and target to the computed row
        rules_matrix(i, end - 1) = weight;
        rules_matrix(i, end) = targets(i);
    end
    
    % Sort row to solve conflicts
    sortrows(rules_matrix, 'descend');
    
    past_row = rules_matrix(1, :);
    unique_rows = false(1, size(features, 1));
    unique_rows(1) = true;
    
    % Select the rule with the maximum weight from a conflict group
    for i = 2 : size(rules_matrix, 1)
    
        actual_row = rules_matrix(i, :);
    
        if ~isequal(past_row(1 : end - 2), actual_row(1 : end - 2))
    
            past_row = actual_row;
            unique_rows(i) = true;
        end
    end
    
    % Remove conflicts and generate the FIS numeric rule descriptions
    filtered_rules_matrix = rules_matrix(unique_rows, :);
    rules = [ ... 
        filtered_rules_matrix(:, 1 : size(features, 2)), ...
        filtered_rules_matrix(:, end), ...
        filtered_rules_matrix(:, end - 1), ...
        ones(size(filtered_rules_matrix, 1), 1) ...
    ];
end