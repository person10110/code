format 
tic
Z = [];
N = [];
N1 = []; 
N2 = [];

N3 = []; 
N4 = []; 
for nn = 1:10000
GoodnessfitScotlandN
Z = [Z; X1 X2 Results2 Results1 Results3];
N = [N; length(find(PP(:,1)<1)) length(find(PP(1:10,1)<1))  length(find(PP(11:20,1)<1)) length(find(PP(21:30,1)<1)) length(find(PP(31:40,1)<1)) length(find(PP(41:50,1)<1))]; 
N1 = [N1; length(find(PPP(:,1)<1)) length(find(PPP(1:10,1)<1))  length(find(PPP(11:20,1)<1)) length(find(PPP(21:30,1)<1)) length(find(PPP(31:40,1)<1)) length(find(PPP(41:50,1)<1))]; 
N3 = [N3; length(find(X(:,2)<0)) length(find(X(:,2)<-5)) length(find(X(:,2)<-10)) length(find(X(:,2)>5)) length(find(X(:,2)>10))]; 
N4 = [N4; length(find(X(:,1)>0)) length(find(X(:,1)>1)) length(find(X(:,1)>2)) length(find(X(:,1)<0))]; 
end

toc

 writematrix(Z, 'Book15.xlsx', 'Sheet', 1);
 writematrix(N, 'Book15.xlsx', 'Sheet', 2);
 writematrix(N1, 'Book15.xlsx', 'Sheet', 3);
 writematrix(N3, 'Book15.xlsx', 'Sheet', 4);
 writematrix(N4, 'Book15.xlsx', 'Sheet', 5);

    

