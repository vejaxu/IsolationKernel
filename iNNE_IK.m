function [feature_map] = iNNE_IK(X_train, X_query, psi, t)
% INNE_IK_MATLAB Implements Isolation Kernel transformation (iNNE-IK).
%
% Inputs:
%   X_train - Data used to construct the kernels (Training data) [n_train x d]
%   X_query - Data to be transformed (Test/Query data) [n_query x d]
%   psi     - Subsampling size (number of centroids per estimator)
%   t       - Ensemble size (number of estimators)
%
% Output:
%   feature_map - Sparse feature matrix [n_query x (t * psi)]
%                 Each row represents a data instance.

    [n_source, ~] = size(X_train);
    [n_query, ~] = size(X_query);

    % Pre-allocate cell array to store features from each iteration.
    % This is faster than growing a matrix inside a loop (ndata = [ndata, new]).
    features_cell = cell(1, t);

    % Pre-calculate column offsets for linear indexing (vectorization trick)
    % This is used to quickly place '1's in the correct positions.
    % Corresponds to: 0, (psi+1), 2*(psi+1), ...
    col_offsets = 0 : (psi + 1) : (n_query - 1) * (psi + 1);

    for i = 1:t
        % --- 1. Sampling (Subsampling centroids) ---
        % Randomly select psi indices without replacement
        sample_idx = datasample(1:n_source, psi, 'Replace', false);
        Centroids = X_train(sample_idx, :);
        
        % --- 2. Remove Duplicates ---
        % If duplicates exist, distances will be 0, messing up radius calc.
        % We keep unique centroids only.
        Centroids = unique(Centroids, 'rows', 'stable');
        % distinct_k is the actual number of unique centroids (<= psi)
        distinct_k = size(Centroids, 1); 
        
        % --- 3. Calculate Adaptive Radius ---
        % Compute distance to the nearest neighbor (excluding self)
        % 'Smallest', 2 returns 2 rows: 
        % Row 1: distance to self (0)
        % Row 2: distance to nearest neighbor
        dist_matrix_self = pdist2(Centroids, Centroids, 'minkowski', 2, 'Smallest', 2);
        Radii = dist_matrix_self(2, :); 
        
        % --- 4. Transform (Find Nearest Centroid for Query Data) ---
        % Find the single nearest centroid for each query point
        [dist_to_center, nearest_idx] = pdist2(Centroids, X_query, 'minkowski', 2, 'Smallest', 1);
        
        % --- 5. Apply Hypersphere Constraint ---
        % Check if distance <= radius of the nearest centroid.
        % Note: nearest_idx contains indices 1 to distinct_k.
        
        % Get the radius corresponding to the nearest centroid for each point
        thresholds = Radii(nearest_idx);
        
        % Identify points falling OUTSIDE the ball
        outliers_mask = dist_to_center > thresholds;
        
        % Assign outlier points to a "garbage bin" index (psi + 1)
        % nearest_idx is now in range [1, psi+1]
        nearest_idx(outliers_mask) = psi + 1;
        
        % --- 6. Construct Sparse Feature Block (Vectorized) ---
        % We want a matrix 'z' of size [(psi+1) x n_query]
        % z(row, col) = 1 where row = nearest_idx(col)
        
        % Using Linear Indexing: Index = Row + (Col-1)*NumRows
        linear_indices = nearest_idx + col_offsets;
        
        % Create sparse matrix directly (Memory Efficient)
        % Arguments: sparse(row_idx, col_idx, value, num_rows, num_cols)
        % We create a vector of all '1's at the calculated linear positions.
        % Note: We create the matrix as Transposed (n_query x (psi+1)) for easier storage,
        % or keep as is. The original code creates (psi+1) x n_query.
        % Let's stick to creating the sparse matrix directly.
        
        % Conceptually: z = zeros(psi+1, n_query); z(linear_indices) = 1;
        % But we skip the zeros step.
        
        % We construct the sparse matrix using (i,j,v) triples.
        % Rows: nearest_idx, Cols: 1:n_query, Values: 1
        z_sparse = sparse(nearest_idx, 1:n_query, 1, psi + 1, n_query);
        
        % Remove the "garbage bin" row (the last row: psi+1)
        % Only rows 1 to psi contain valid features.
        % If distinct_k < psi, rows (distinct_k+1) to psi will naturally be empty (zeros).
        z_sparse(psi + 1, :) = []; 
        
        % Store the transposed result (n_query x psi) to match Python output style
        features_cell{i} = z_sparse'; 
    end

    % --- 7. Concatenate All Features ---
    % Combine all blocks horizontally: [Block1, Block2, ..., BlockT]
    % Result size: n_query x (t * psi)
    feature_map = [features_cell{:}];
    
end