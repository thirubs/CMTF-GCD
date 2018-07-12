# CMTF-GCD

CMTF GCD


Initialize the following variables and call the CMTF_GCD function. 


Example:
% [Input]
% X           data tensor                        (sptensor)
% Y           data matrix
% J and JM    the nRankf of Tensor and Matrix             (double)
% U and UM        Initialization of factor matrices  (cell)
% maxiter     the numer of iterations            

[U_GCD,var_GCD,grad_GCD,varA_updated_GCD] = CMTF_GCD(X,J,U,Y,JM,UM,maxiter);
