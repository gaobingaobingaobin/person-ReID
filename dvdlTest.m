function rank=dvdlTest(D,galleryTestingFeatures,galleryTestingId,probeTestingFeatures,probeTestingId,eta)

Yg=avgFeatures(galleryTestingFeatures,galleryTestingId);
Yp=avgFeatures(probeTestingFeatures,probeTestingId);
m=size(Yg,2);n=size(Yp,2);assert(m==n);

% Compute sparse codes
for i=1:n
    sc_probe{i}=fistaTest(D,Yp(:,i),eta);
    sc_gallery{i}=fistaTest(D,Yg(:,i),eta);
end

% Match
for i=1:n
    currx=sc_probe{i};
    dist{i}=zeros(1,n);
    for j=1:n
        curry=sc_gallery{j};
        dist{i}(j)=norm(currx-curry);        
    end

end

rank=zeros(n,1);
for i=1:n
    [~,ind1]=sort(dist{i},'ascend');
    r = find(ind1 == i);
    rank(r) = rank(r) + 1;
end

rank = cumsum(rank) / n * 100;

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

function xk = fistaTest(A,b,lamReq)

xk=A\b;yk=xk;xkm1=xk;tkm1=1;tk=1;

shrink=@(x,fac)sign(x) .* max(abs(x) - fac,0);

beta=1.5; L0=1;L=L0;eta=0.95; 
lambda0=0.5*L0*norm(A'*b,inf);
lam=lambda0;lambda_bar=lamReq;
max_iter=20000;

iter=0;
while(iter<max_iter)
    iter=iter+1;
    yk=xk+((tkm1-1)/tk)*(xk-xkm1);
    g=A'*(A*yk-b);
    stop_backtrack=0;
    while(~stop_backtrack)
        temp=yk-(1/L)*g;
        xkp1=shrink(temp,lam/L);
        temp1 = 0.5*norm(b-A*xkp1)^2 ;
        temp2 = 0.5*norm(b-A*yk)^2 + (xkp1-yk)'*g + (L/2)*norm(xkp1-yk)^2 ;
        if temp1 <= temp2
            stop_backtrack = 1 ;
        else
            L = L*beta ;
        end
    end
    lam=max(eta*lam,lambda_bar);
    tkp1 = 0.5*(1+sqrt(1+4*tk*tk)) ;
    tkm1=tk;
    tk=tkp1;
    xkm1=xk;
    xk=xkp1;
end

