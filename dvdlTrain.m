function [D,X,Y,errs]=dvdlTrain(galleryTrainingFeatures,galleryTrainingId,probeTrainingFeatures,probeTrainingId,eta)

P=avgFeatures(galleryTrainingFeatures,galleryTrainingId);
Q=avgFeatures(probeTrainingFeatures,probeTrainingId);

lam1=eta;lam2=eta;lam3=eta;
nIters=2;
% nIters=1;
N=size(P,2);N1=size(Q,2);M=size(P,1);
K=N+N1;
D=rand(M,K)-0.5;D=D-repmat(mean(D,1), size(D,1),1);D=D*diag(1./sqrt(sum(D.*D)));
X=rand(K,N)-0.5;X=X-repmat(mean(X,1), size(X,1),1);X=X*diag(1./sqrt(sum(X.*X)));
Y=rand(K,N1)-0.5;Y=Y-repmat(mean(Y,1), size(Y,1),1);Y=Y*diag(1./sqrt(sum(Y.*Y)));

for i=1:nIters
        X=getSparseCodes(P,D,Y,lam1,lam3);
        Y=getSparseCodes(Q,D,X,lam2,lam3);
    
    
    currD=D;
        D=learnBasis(P,Q,X,Y);
%     D=learnBasis(P,Q,X,Y,D);
    newD=D;
    errs(i,1)=norm(P-D*X,'fro');errs(i,2)=norm(Q-D*Y,'fro');
    errs1(i)=errs(i,1)+errs(i,2);
end

function [f,uids] = avgFeatures(feat,ids)

uids=unique(ids,'stable');
f=[];
for i=1:length(uids)
    currId=uids(i);
    ids1=find(ids==currId);
    tmp=zeros(size(feat,1),1);
    for j=1:length(ids1)
        tmp=tmp+feat(:,ids1(j));
    end
    tmp=tmp./length(ids1);
    f=[f tmp];
end

function D=learnBasis(P,Q,X,Y)%250*150,250*150,300*150,300*150
% function D=learnBasis(P,Q,X,Y,D)
% MaxIteration_num=5;
MaxIteration_num=1;
beta = 0.002;
D=(P*X'+Q*Y')*pinv(X*X'+Y*Y');
%  D=(P*X'+Q*Y')*pinv(X*X'+Y*Y');%类似初始化 250*300
%% gaobin added 正交字典约束下，采用yangmeng2013joint JDDLDR
numcomps = size(D,1);
iter_num_sub= 1;
while iter_num_sub<MaxIteration_num % subject iteration for updating p, iteration ends when the contiguous function get close enough or reach teh max_iteration
    phi = (X-D'*P)*(X-D'*P)'+(Y-D'*Q)*(Y-D'*Q)'; %300*300
    [U,V] = eig(phi); p1 = U(:,1:numcomps);
      D = D+beta*(p1'-D);
    iter_num_sub=iter_num_sub+1;
end
% Fix alpha, update D. begin
% p=size(D,2);
% alpha1=X;
% alpha2=Y;
% for i=1:p
%     ai1        =    alpha1(i,:);
%     ai2        =    alpha2(i,:);
%     Yi1         =    P-D*alpha1+D(:,i)*ai1;
%     Yi1         =    Q-D*alpha2+D(:,i)*ai2;
%     di        =    Yi1*ai1'+Yi1*ai1';
%     di        =    di./norm(di,2);
%     D(:,i)    =    di;
% end
% Fix alpha, update D. end

function X=getSparseCodes(P,D,Y,lam1,lam2)
X=[];
for i=1:size(Y,2)
    i
    ysame=Y(:,i);
    Yt=Y;
    Yt(:,i)=[];
    X(:,i)=getSparseCodes1(P(:,i),D,ysame,Yt,lam1,lam2);
end

function x=getSparseCodes1(p,D,ys,yd,lam1,lam2)
n=size(ys,1);
cvx_begin quiet
variable x(n)
minimize (norm(p-D*x)+lam1*norm(x,1))
subject to
norm(x-ys)<=0.1;
for i=1:size(yd,2)
    norm(x-yd(:,i))<=100;
end
cvx_end
