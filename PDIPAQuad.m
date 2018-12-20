function [output, x,lambda, nu] = PDIPAQuad(Q,c,A,b,tol)
%Implements the primal dual interior point (path-following) algorithm for
%solving the quadratic program of the form:  minimize 1/2x'Qx+c'x
%                                            subject to Ax=b (can add
%                                            slacks if not in this form)
%                                                        x>=0
%Q is NxN
%x is Nx1 decision variables
%c is Nx1
%A is MxN constraints
%b is Mx1

N=length(c); %Number of decision variables
M=size(A,1); %Number of constraints

%Initialization
x=ones(N,1);
lambda=ones(N,1);
nu=ones(M,1);
Theta=[x;lambda;nu]; %Initial Point
maxiter=1000;

%Reform
X=diag(x);
e=ones(N,1);
Lambda=diag(lambda);

%Calculate the residuals
rho=A*x-b; %primal residual;
sigma=A'*nu-lambda+c+Q*x; %dual residual;
gamma=x'*lambda; %complementarity
mu=gamma/(5*N);


%Measure residuals
m1=norm(rho,1); %measure of primal constraint
m2=norm(sigma,1); %measure of dual constraint
m3=norm(gamma,1); %measure of completmentarity
err=max([m1 m2 m3]);
iter=0;

while (err>tol)
    %Break if hit maximum number of iterations
    iter=iter+1;
    if iter>=maxiter
        disp('Maximum number of iterations has been reached.')
        break;
    end
    
    %Create linear system of equations DF DelT = -F
    DF=[Q -eye(N) A'; Lambda X zeros(N,M); A zeros(M,N) zeros(M,M)];
    F=[A'*nu-lambda+c+Q*x;X*lambda-mu*e;A*x-b];
    DelT=DF\(-F);
    
    %Determine the change in each of the variables.
    Delx=DelT(1:N,1);
    Dellambda=DelT(N+1:2*N,1);
    Delnu=DelT(2*N+1:2*N+M,1);
    
    mm = max([-Delx./x; -Dellambda./lambda]); %-Delnu./nu;
    step(iter)   =  min(0.9/mm,1);
    
    if step(iter) < 10^(-8)%if the step size is too small stop
        iter
        disp('Step size is too small')
        break
    end
    
    %Update solution
    ThetaNew=Theta+step(iter)*DelT; %Update Primal
    x=ThetaNew(1:N,1);
    lambda=ThetaNew(N+1:2*N,1);
    nu=ThetaNew(2*N+1:2*N+M,1);
    Theta=ThetaNew;
    X=diag(x);
    Lambda=diag(lambda);
    
    %Calculate the residuals
    rho=A*x-b; %primal residual;
    sigma=A'*nu-lambda+c+Q*x; %dual residual;
    gamma=x'*lambda; %complementarity
    
    %Update mu
    mu=gamma/(5*N);
    
    %Measure residuals
    m1(iter)=norm(rho,1); %measure of primal constraint
    m2(iter)=norm(sigma,1); %measure of dual constraint
    m3(iter)=norm(gamma,1); %measure of completmentarity
    err=max([m1(iter) m2(iter) m3(iter)]);
    
    output.iteration=iter;
    output.step(iter)=step(iter);
    output.complementarity=m1(iter);
    output.dual_feasibility=m2(iter);
    output.primal_feasibility=m3(iter);
    output.minvalue=1/2*x'*Q*x+c'*x;
    output.bounded=c'*x+nu'*b+x'*Q*x; %duality gap
    output.xx(:,iter)=x;
    output.nunu(:,iter)=nu;
    output.lambdalambda(:,iter)=lambda;
    
end


end

