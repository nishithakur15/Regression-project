data=LINTHALL;
data=removevars(data,{'Obs','Loc','Type'});
Y=data.BIO;
X=data{:,2:end};
predictor_names=data.Properties.VariableNames(2:end);
%%Ordinary Least Squares
X_ols=[ones(size(X,1),1),X];
[b_ols,~,res,~,stats]=regress(Y,X_ols);
fprintf("\n=====OLS RESULTS=====\n");
fprintf("R^2=%.4f\n",stats(1));
fprintf("F-statistic p-value=%.4f\n",stats(3));
SSE_OLS=sum(res.^2);
fprintf("SSE_OLS=%.4f\n",SSE_OLS);
%%Computing Standard Errors
yhat=X_ols*b_ols;
residuals=Y-yhat;
n=length(Y);
p=size(X_ols,2);
sigma2=sum(residuals.^2)/(n-p);
covB=sigma2*inv(X_ols'*X_ols);
SE_OLS=sqrt(diag(covB));
SE_sum_OLS=sum(SE_OLS);
fprintf("Sum of standard errors (OLS)=%.4f\n\n",SE_sum_OLS);
%%Collinearity
fprintf("=====COLLINEARITY=====\n");
VIF=diag(inv(corrcoef(X)));
disp("VIF values:");
disp("VIF values:",VIF);
disp(array2table(VIF','VariableNames',predictor_names));
corr_matrix=corrcoef(X);
figure;heatmap(predictor_names,predictor_names,corr_matrix);
title('Correlation Matrix');
[Q,R]=qr(zscore(X),0);
E=eig(inv(R'*R));
cond_index=sqrt(max(E)./E);
fprintf("\nCondition Indices:\n");
disp(cond_index);
%%Principal Component Regression
Z=zscore(X);
[coeff,score,latent,~,expalined]=pca(Z);
figure;
pareto(explained);
pareto(expalined);
%made a spelling error, it should be explained instead of expalined
title('Scree Plot-Variance Explained');
cum_var=cumsum(expalined);
disp('Cumulative variance explained by PCs:');
disp(cum_var);
k=find(cum_var>95,1);
fprintf("Using %d principal components.\n",k);
PC_reduced=score(:,1:k);
X_pcr=[ones(size(X,1),1),PC_reduced];
[b_pcr,~,res_pcr]=regress(Y,X_pcr);
SSE_PCR=sum(res_pcr.^2);
fprintf("\n=====PCR RESULTS=====\n");
fprintf("SSE_PCR=%.4f\n",SSE_PCR);
beta_original=coeff(:,1:k)*b_pcr(2:end);
fprintf("\nPCR coefficients converted to original scale:\n"
fprintf("\nPCR coefficients converted to original scale:\n");
disp(array2table(beta_original','VariableNames',predictor_names));
%%Computing Standard Errors
covB_pcr=sigma2*inv(X_pcr'*X_pcr);
SE_PCR=sqrt(diag(covB_pcr));
SE_sum_PCR=sum(SE_PCR);
fprintf("Sum of standard errors (PCR)=%.4f\n\n",SE_sum_PCR);
%%%% PART II OF THE PROJECT %%%%
data=LINTH_5;
data=removevars(data,{'Obs','Loc','Type'});
Y=data.BIO;
X=data{:,{'SAL','pH','K','Na','Zn'}};
data.Properties.VariableNames
mean(data.Var8)
mean(data.Var9)
mean(data.Var10)
mean(data.Var6)
clear data
clear LINTH_5
data=LINTH_5;
data=removevars(data,{'Obs','Loc','Type'});
Y=data.BIO;
X=data{:,{'SAL','pH','K','Na','Zn'}};
%% We have had to do this again since for the previous code, MATLAB did not import the data correctly and other tables popped up due to misreading of the data
%% Collinearity
R=corrcoef(X);
VIF=diag(inv(R));
disp('VIF values:');
disp(array2table(VIF','VariableNames',predictor_names));
size(X)
predictor_names
%% This error was because we did not change the predictor names.
predictor_names = {'SAL','pH','K','Na','Zn'};
disp('VIF values:');
disp(array2table(VIF','VariableNames',predictor_names));
[Q,Rmat]=qr(zscore(X),0);
E=eig(inv(Rmat'*Rmat));
cond_index=sqrt(max(E)./E);
disp('Condition indices:');
disp(cond_index);
figure;heatmap(predictor_names,predictor_names,R);
title('Correlation Matrix');
%% Regression (Stepwise)
stepmodel = stepwiselm(data, 'linear', ...
'ResponseVar','BIO', ...
'PredictorVars', predictor_names, ...
'Upper','linear','Lower','constant', ...
'PEnter',0.15,'PRemove',0.15);
stepmodel = stepwiselm(data, 'linear', ...
'ResponseVar','BIO', ...
'PredictorVars', predictor_names, ...
'Upper','linear','Lower','constant', ...
'PEnter',0.15, 'PRemove',0.20);
disp(stepmodel);
vars=stepmodel.PredictorNames;
X2=data{:,vars};
R2=corrcoef(X2);
VIF2=diag(inv(R2));
disp('VIF after stepwise:');
disp(array2table(VIF2','VariableNames',vars));
%% Subset Selection
comb=nchoosek(1:5,2);
AIC_vals=[];
BIC_vales=[];
SSE_vals=[];
for i=1:size(comb,1)
idx=comb(i,:);
vars=predictor_names(idx);
tbl=data(:,[{'BIO'},vars]);
m=fitlm(tbl,'linear');
AIC_vals(i)=m.ModelCriterion.AIC;
BIC_vals(i)=m.ModernCriterion.BIC;
%% made a mistake with spelling, would have to type again
end
for i = 1:size(comb,1)
idx = comb(i,:);
vars = predictor_names(idx);
tbl = data(:, [{'BIO'}, vars]);
m = fitlm(tbl,'linear');
AIC_vals(i) = m.ModelCriterion.AIC;
BIC_vals(i) = m.ModelCriterion.BIC;
SSE_vals(i) = sum(m.Residuals.Raw.^2);
end
results = table(comb, AIC_vals', BIC_vals', SSE_vals');
disp(results);
%%Ridge Regression and Ridge Trace
X_std=zscore(X);
Y_std=zscore(Y);
lambdas=0:0.1:20;
coeffs=zeros(length(lambdas),5);
for i = 1:length(lambdas)
lambda = lambdas(i);
coeffs(i,:) = ridge(Y_std, X_std, lambda, 0);
end
for i = 1:length(lambdas)
lambda = lambdas(i);
b = ridge(Y_std, X_std, lambda, 0);   % returns 6 coefficients
coeffs(i,:) = b(2:end);              % keep only SAL, pH, K, Na, Zn
end
figure;
plot(lambdas,coeffs);
xlabel('\lambda');ylabel('Coefficients');
legend({'SAL','pH','K','Na','Zn'});
title('Ridge Trace');
keep={'SAL','pH'};
X_ridge=data{:,keep};
R3=corrcoef(X_ridge);
VIF3=diag(inv(R3));
disp('VIF after ridge variable selection:');
disp(array2table(VIF3','VariableNames',keep));