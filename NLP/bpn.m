function prclass = bpn(input_matrix,output_matrix,training_input_data,training_output_data,test_input_data,test_output_data)
l = length(input_matrix(1,:)) 		     %no of neurons in input layer
m = l   					     %no of neurons in hidden layer , it can lie between l to 2*l
n =  length(output_matrix(1,:)); 		     %no of neurons in output layer
v = zeros(l,m); 	     %weight vector between input and hidden layer
w = zeros(m,n); 	     %weight vector between hidden and output layer
delta_v = zeros(l,m); %weight change matrix for v
delta_w = zeros(m,n); %weight change matrix for w
alpha = 1;
learning_rate = 0.3;

%assign random values to v and w matrix
for i =1:l 
	for k = 1:m
		v(i,k) = (1).*rand(1,1);
	end
end

for i = 1:length(w) 
	for k = 1:length(w(1))
		w(i,k) = (1).*rand(1,1);
	end
end


ee=0;
for iter = 1:length(training_input_data) 
%	len = length(training_input_data(i));
	Ii  = (training_input_data(iter,:));    %input pattern
  %  fprintf('Ii: row = %d col= %d\n',length(Ii),length(Ii(1,:)));
	Oi  = Ii 	;				 %output of input linear , since linear transfer function
  %  fprintf('Oi: row = %d col= %d\n',length(Oi),length(Oi(1,:)));
	Ih = transpose(v) * transpose(Oi); 			 %calulating input to hidden layer
   % fprintf('Ih: row = %d col= %d\n',length(Ih),length(Ih(1,:)));
	Oh = 1./(1+exp(-1*Ih)); 			 %Output from hidden layer , since sigmoidal function is used
    %fprintf('Oh: row = %d col= %d\n',length(Oh),length(Oh(1,:)));
	Io = transpose(w) * Oh ;  %calculating input for output layer
     %fprintf('Io: row = %d col= %d\n',length(Io),length(Io(1,:)));
   % fprintf('v: row = %d col= %d\n',length(v),length(v(1,:)));
   % fprintf('w: row = %d col= %d\n',length(w),length(w(1,:)));
    %display(length(Io));
    %display(length(Io(1)));
 	Oo = 1./(1+exp(-1*Io)); 			 %calculating output of output layer
    %fprintf('Oo: row = %d col= %d\n',length(Oo),length(Oo(1,:)));

	%now we need to calculate error , and adjust weights accordingly
	E=0;
	for j = 1:n 
	  E = E+ ((training_output_data(iter,j)-Oo(j))^2);
	end
	ee = ee+E;
	%let us adjust weights
	d = (training_output_data(iter)-Oo)*(Oo)*(1-Oo);
   %  fprintf('d: row = %d col= %d\n',length(d),length(d(1,:)));
	Y = Oh * d(1); 				 			 
	delta_w = (alpha * delta_w )+ (learning_rate * Y) ;	
	e = w * d(1);
    eoh = e*transpose(Oh);
  %  fprintf('e: row = %d col= %d\n',length(e),length(e(1,:)));
  %  fprintf('eoh: row = %d col= %d\n',length(eoh),length(eoh(1,:)));
	d_star = (eoh)*(1-Oh) ;
  %  fprintf('d_star: row = %d col= %d\n',length(d_star),length(d_star(1,:)));
	X = (Oi) * (d_star);
	delta_v = (alpha * delta_v) + (learning_rate * X);
	v = v + delta_v	;			 %Updating V vector
	w = w + delta_w	;			 %Updating W vector
	
end

%Let us find error rate with test data
E=0;
prclass=[];
length(test_output_data)
for iter = 1:400
	
    Ii  = (test_input_data(iter,:));    %input pattern
   % fprintf('Ii: row = %d col= %d\n',length(Ii),length(Ii(1,:)));
	Oi  = Ii 	;				 %output of input linear , since linear transfer function
   % fprintf('Oi: row = %d col= %d\n',length(Oi),length(Oi(1,:)));
	Ih = transpose(v) * transpose(Oi); 			 %calulating input to hidden layer
   % fprintf('Ih: row = %d col= %d\n',length(Ih),length(Ih(1,:)));
	Oh = 1./(1+exp(-1*Ih)); 			 %Output from hidden layer , since sigmoidal function is used
    %fprintf('Oh: row = %d col= %d\n',length(Oh),length(Oh(1,:)));
	Io = transpose(w) * Oh ;  %calculating input for output layer
    % fprintf('Io: row = %d col= %d\n',length(Io),length(Io(1,:)));
    %fprintf('v: row = %d col= %d\n',length(v),length(v(1,:)));
    %fprintf('w: row = %d col= %d\n',length(w),length(w(1,:)));
    %display(length(Io));
    %display(length(Io(1)));
 	Oo = 1./(1+exp(-1*Io)); 			 %calculating output of output layer
    %fprintf('Oo: row = %d col= %d\n',length(Oo),length(Oo(1,:)));
    
	prclass(iter) = Oo;
	  E = E+ ((test_output_data(iter)-Oo)^2);
	
	
	%E = E + e;
end
display(E)
