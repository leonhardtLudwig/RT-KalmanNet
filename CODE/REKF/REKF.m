function [Xrekf,A,C,G,th,a,b,V]=REKF(x0,y,V0,B,D,f,h,sym_x,c,T)
Q=B*B';
R=D*D';
n=size(Q,1);
p=size(R,1);
Xrekf=zeros(n,T+1);
Xrekf(:,1)=x0;
Xn=zeros(n,T);
a=zeros(n,T+1);
b=zeros(p,T);
V=zeros(n,n,T+1);
V(:,:,1)=V0;
A=zeros(n,n,T);
C=zeros(p,n,T);
G=zeros(n,p,T);
th=zeros(1,T);
for i=1:T
    %C_t
    h_l=jacobian(h,sym_x);
    C(:,:,i)=subs(h_l,sym_x,Xrekf(:,i));
    %b_t
    b(:,i)=subs(h,sym_x,Xrekf(:,i))-C(:,:,i)*Xrekf(:,i);
    %L_t
    L=V(:,:,i)*C(:,:,i)'*inv(C(:,:,i)*V(:,:,i)*C(:,:,i)'+R); 
    %h(\hat x_t,u_t)
    hn= subs(h,sym_x,Xrekf(:,i));
    hn = eval(hn);
    %\hat x_t|t
    Xn(:,i)=Xrekf(:,i)+L*(y(:,i)-hn);
    %A_t
    f_l=jacobian(f,sym_x);
    A(:,:,i)=subs(f_l,sym_x,Xn(:,i));
    %G_t
    G(:,:,i)=A(:,:,i)*L;
    %\hat x_t+1
    Xrekf(:,i+1)=subs(f,sym_x,Xn(:,i));    
    %a_t
    a(:,i)=subs(f,sym_x,Xn(:,i))-A(:,:,i)*Xn(:,i);
    %V_t+1
    P=A(:,:,i)*V(:,:,i)*A(:,:,i)'-A(:,:,i)*V(:,:,i)*C(:,:,i)'*inv(C(:,:,i)*V(:,:,i)*C(:,:,i)'+R)*C(:,:,i)*V(:,:,i)*A(:,:,i)'+Q;
    %th_t
    value=1;
    t1=0; 
    e = eig(P);
    r = max(abs(e));
    t2=(1-10^-5)*(r)^-1;
    while abs(value)>=10^-9
       t=0.5*(t1+t2);
       value=trace(inv(eye(n)-t*P)-eye(n)) + log(det(eye(n) -t*P))-c;
       if value>0
            t2=t;
       else
            t1=t;
       end
    end
    th(i) = t;
    V(:,:,i+1)=inv(inv(P)-th(i)*eye(n));
end

