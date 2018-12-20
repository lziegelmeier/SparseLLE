function [Y,W] = SparseLLE(X,K,d,lambda)

%Sparse Locally Linear Embedding algorithm detailed in article
%L. Ziegelmeier, M. Kirby, C. Peterson, "Sparse Locally Linear Embedding,"
%Procedia Computer Science, 108C (2017), pages 635-644.

%Code by Lori Ziegelmeier, 
%NOTE: this code has not been optimized, and should be written in a more
%vectorized format

%Input:
% X = Dxp data set X
% K = the number of nearest neighbors
% d=dimension of lower-dimensional embedding space 
% lambda = parameter to enforce neighbor selection

%Outputs: 
% Y = dxp lower-dimensional embedding
% W = Kxp weight matrix with sparsity induced


S=size(X);
p = S(2) %Number of points
D= S(1) %Dimensionality

%%Nearest Neighbor Computation
disp('Computing Nearest Neighbors')

X2 = X' * X;
X2DiagRep = repmat(diag(X2), 1, p);
Distance = X2DiagRep' + X2DiagRep - 2 * X2;
[SortDist, Index] = sort(Distance); %Sorts each row of the distance matrix

%Must use when there are repeated points (this should be optimized)                                                    
for j=1:p
    count=0;
    for i=1:p
        if i==1 
            count=count+1;
        elseif SortDist(i,j)==0
            count=count+1;
        end
    end
    NeighborInd(:,j)=Index(count+1:count+K,j);
end

%Creating a data cube containing the matrix of neighborhoods for each
%point Xi, N(:,:,i) is the matrix of neighboring points for point Xi
N=zeros(D,K,p);
for i=1:p
    Ni=zeros(D,K);
    for t=1:K
        Ni(:,t)=X(:,NeighborInd(t,i));
    end
    N(:,:,i)=Ni;
end
N;

disp('Done Computing Nearest Neighbors')



%%Determining weights
%Computing the Reconstruction Weights by solving a quadratic program
%Decision variables are the nonzero entries of W, wtilde, with
%wtildeplus and wtildeminus
%MATLAB standard form for a quadratic program is 1/2x'Hx+f'x
%subject to Ax<=b, Aeqx=beq, lb<=x<=ub

%%%Computing the Sparse Weights
disp('Computing the Sparse Weights Using PDIP')

S=size(X);
p = S(2); %Number of points

K=size(N,2); %Number of nearest neighbors

H=zeros(2*K,2*K,p);
f=zeros(2*K,p);
for i=1:p
    Ni=N(:,:,i); %Forming a matrix of all neighbors of the point Xi
    Hhat=zeros(K,K);
    ftilde=zeros(K,1);
    fhat=zeros(K,1);
    Xi=X(:,i);
    for j=1:K
        Xj=Ni(:,j);
        ftilde(j)=-2*Xi'*Xj+lambda*(norm(Xi-Xj,2));%-norm(Xi-X12,2)+0.01)^2;
        fhat(j)=2*Xi'*Xj+lambda*(norm(Xi-Xj,2));%-norm(Xi-X12,2)+0.01)^2;
        for l=1:K
            Xl=Ni(:,l);
            Hhat(j,l)=Xj'*Xl;
        end
    end
    Htilde=zeros(2*K,2*K);
    Htilde=[Hhat -Hhat; -Hhat Hhat];
    H(:,:,i)=Htilde;
    f(:,i)=[ftilde; fhat];
end


%Solving the quadratic program for each point Xi
Aeq=[ones(1,K) -ones(1,K)];
b=1;    
WtildeMatrix=zeros(K,p);
fval=0;
for i=1:p
    Htilde=H(:,:,i);
    ftilde=f(:,i);    
    [output, x,lambdaout, nu] = PDIPAQuad(2*(1-lambda)*Htilde,ftilde,Aeq,b,10^-6); 
    Wtildeplus=x(1:K);
    Wtildeminus=x(K+1:2*K);
    Wtilde=Wtildeplus-Wtildeminus;
    WtildeMatrix(:,i)=Wtilde;

    fval=fval+output.minvalue;
end

disp('Done Computing Weights')


%%%Computing the Lower Dimension Embedded Vectors
disp('Computing Embedding Vectors')

M = sparse(1:p,1:p,ones(1,p),p,p,4*K*p); 
for i=1:p
   w = W(:,i);
   %w(find(w<10^(-5)))=0;
   j = NeighborInd(:,i);
   M(i,j) = M(i,j) - w';
   M(j,i) = M(j,i) - w;
   M(j,j) = M(j,j) + w*w';
end;

options.disp = 0; options.isreal = 1; options.issym = 1; 
[eigenvec, eigenvals]=eigs(M,d+1,0,options); %Computes the d+1 e-vals and e-vect of M that are closest to 0
[SortEVals,SortEValIndex]=sort(diag(eigenvals));
SortEVecs=eigenvec(:,SortEValIndex);
Y2=SortEVecs(:,2:d+1)';
Y=Y2*sqrt(p); 

disp('Done Computing Embedding Vectors')

end

