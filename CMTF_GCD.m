function [U, total_var_updates, total_grad_updates,updated_variable] = CMTF_GCD(X,J,U,Y,JM,UM,maxiters)
%function [U,output] = CMTD_GCD(X,J,U,Y,JM,UM,maxiters)
%%CMTF_GCD. Nonnegative Coupled Matrix Tensor Factorization with Greedy Coordinate Descent
% This implementation can be used for only 3-order tensor with 1 matrix
% shared in mode1. 

% Note: You should have Matlab tensor toolbox library to run this.
%% Citation:
% If you use this code, cite any of the following paper
% 1. Balasubramaniam, Thirunavukarasu, Nayak, Richi, & Yuen, Chau. "Nonnegative coupled matrix tensor factorization for smart city spatiotemporal pattern mining." In International Conference on Machine Learning, Optimization, and Data Science,  pp. 520-532. Springer, Cham, 13-16 September 2018, Pisa, Italy 2018. 

% 2. Balasubramaniam, Thirunavukarasu, Nayak, Richi, & Yuen, Chau. "People to people recommendation using coupled nonnegative Boolean matrix factorization. In IEEE International Conference on Soft-Computing and Network Security, 14-16 February 2018, Coimbatore, India. 


%%
% [Input]
% X           data tensor                        (sptensor)
% Y           data matrix
% J and JM    the nRankf of Tensor and Matrix             (double)
% U , UM          Initialization of factor matrices  (cell)
% maxiters     the numer of iterations            
% [Output]
% U        the factorization result             (ktensor)

%%
% Execution Examples
% [U] = NCPHALS_sample(X,J)
%% Title
fprintf('CMTF_GCD\n');

%% Initialization for parameters 
%tic
N = ndims(X);
normX = norm(X);
epsilon=1e-12;
tol=1.0e-4;

%% compute Hadamard products for tensor factors
Hmat=sparse(ones(J,J));
for n=1:N
    Hmat= Hmat .* ((U{n,1}' *U{n,1})+epsilon);
end

%% iterations
%tic
time = 0;
total_var_updates = zeros(maxiters,2);
total_grad_updates = zeros(maxiters,2);
for i = 1:maxiters
total_var_updates(i,1) = i;
total_grad_updates(i,1) = i;
   for n= 1:N
    factor = n;
    rows_and_cols = size(U{n});
    rows = rows_and_cols(1);
    cols = rows_and_cols(2);
  	%Update Hadamard matrix for nth mode
	Hmat = Hmat ./ ((U{n,1}' *U{n,1})+epsilon);
    tic
    tmpmat=mttkrp(X,U,n);
	grad= -(tmpmat-(U{n,1} *Hmat));
    F2_grad = zeros(rows,JM);
    HHT=sparse(ones(J,J));
    type = 0;
    if n == 1
        HHT = (UM{2,1}' *UM{2,1})+epsilon;
        
        F2_grad = -((Y*UM{2,1})-(U{1,1} *HHT)); %gradient of input matrix
        F2 = F2_grad;
        grad = grad + F2_grad;
        Hmat_a = Hmat+HHT;      
        tic
        [Unew, var_updates_n, grad_updates_n,updated_variable]=goWiter_NN(grad,Hmat_a,U{n,1},tol,rows,cols,J);
        time = time+ toc;
    else
        tic
        [Unew, var_updates_n, grad_updates_n,~]=goWiter_NN(grad,Hmat,U{n,1},tol,rows,cols,J);
        time = time + toc;
    end
	U{n,1}=  Unew;
    if n == 1
        UM{1,1} = Unew;
        %Check nonnegativity 
	UM{1,1}(UM{1,1}<=epsilon)=epsilon;
        
	%	UM{1,1}=normalize_factor(UM{1,1},2);
	
    end
	%Check nonnegativity 
	U{n,1}(U{n,1}<=epsilon)=epsilon;
	%Normalization (if you need)
	if (n~=N)
%		U{n,1}=normalize_factor(U{n,1},2);
	end
	%Update Hadamard matrix with updated matrix			
	Hmat = Hmat .* ((U{n,1}' *U{n,1})+epsilon);
    
    total_var_updates(i,2) = total_var_updates(i,2) + var_updates_n;
    total_grad_updates(i,2) = total_grad_updates(i,2) + grad_updates_n;
   end 
   
   % Update H of input matrix's factor
   ATA = (U{1,1}' *U{1,1})+epsilon;
   grad_H = (UM{2,1} *ATA)-(Y'*U{1,1});
 
    rows = rows_and_cols(1);
    cols = rows_and_cols(2);
    type = 1;
    tic
   [UMnew, var_updates_n, grad_updates_n,~]=goWiter_NN(grad_H,ATA,UM{2,1},tol,rows,cols,J^2);
   time = time + toc;
   UM{2,1} = UMnew;
   %Check nonnegativity 
	UM{2,1}(UM{2,1}<=epsilon)=epsilon;
	%Normalization (if you need)
	%if (2~=N)
%		UM{2,1}=normalize_factor(UM{2,1},2);
	%end
    %toc
    total_var_updates(i,2) = total_var_updates(i,2) + var_updates_n;
    total_grad_updates(i,2) = total_grad_updates(i,2) + grad_updates_n;
end
