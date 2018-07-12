function [Wout,no_of_var_updates,no_of_gradient_updates, updated_variable] = goWiter_NN(GW,HH,W,tol,n,k,initer)
%% Input Parameters
% GW - Gradient (X(A kathrirao of B))
% HH - Hadamard Product
% W - Factor Matrix to update
% tol - 0.001
% n - number of rows in W
% k - number of columns in W
% initer - maximum number of inner iterations
	% initial maximum function value decreasing over all coordinates. 
	init = 0; 
	%Diagonal of HH
	%double *HH_d = (double *)malloc(sizeof(double)*k);
    %HH_d = 0;
	%for i=1:k
	%	HH_d(i) = HH(i*k+i);
    %end
    %HH_d = diag(HH);
	% Create SWt : store step size for each variables 
	%double *SWt = (double *)malloc(sizeof(double)*k);
    SWt = 0;
	%Get init value - Try to implement the parallelized MTTKRP along with 
    %this initialization by dividing into block of rows 
    %nowidx=0 ;
   updated_variable = zeros(n,k); 
	for i = 1 : n
        
		for  j= 1:k 
            %nowidx = nowidx+1;
			%double s = GW[nowidx]/HH_d[j];
            s = GW(i,j)/HH(j,j);
			s = W(i,j)-s;
			if( s< 0)
				s=0;
            end
			s = s-W(i,j);
			%double diffobj = (-1)*s*GW[nowidx]-0.5*HH_d[j]*s*s;
            diffobj = (-1)*s*GW(i,j)-0.5*HH(j,j)*s*s;
			if ( diffobj > init )
				init = diffobj;
            end
         end
    end

%%stopping condition

%%coordinate descent 
no_of_var_updates = 0;
no_of_gradient_updates = 0;
updated_variable = zeros(i,j);
	for p=1:n
		GWp(p,:) = GW(p,:);
		Wp(p,:) = W(p,:);
		for winner = 1 :initer
			% find the best coordinate 
			 q = -1;
			 bestvalue = 0;

			for i=1:k
				ss = GWp(p,i)/HH(i,i);
				ss = Wp(p,i)-ss;
				if (ss < 0)
					ss=0;
                end
				ss = ss-Wp(p,i);
				SWt(p,i) = ss;
				%double diffobj = (-1)*(ss*GWp[i]+0.5*HH_d[i]*ss*ss);
                diffobj = (-1)*(ss*GWp(p,i)+0.5*HH(i,i)*ss*ss);
				if ( diffobj > bestvalue ) 
					bestvalue = diffobj;
					q = i;
                end
            end
			if ( q==-1 )
				break;
            end

			Wp(p,q) = Wp(p,q)+SWt(p,q);
            W(p,q) = Wp(p,q);
            no_of_var_updates = no_of_var_updates+1;
            updated_variable(p,q) = updated_variable(p,q)+1;
			for i=1:k
				GWp(p,i) = GWp(p,i)+( SWt(p,q)*HH(q,i));
                GW(p,i) = GWp(p,i);
                no_of_gradient_updates = no_of_gradient_updates+1;
            end
			if ( bestvalue < init*tol)
				break;
            end
        end
    end
    
    %for i=1:n*k 
	%	Wout(i) = W(i);
    %end
    Wout = W;


	%free(HH_d);
	%free(SWt);