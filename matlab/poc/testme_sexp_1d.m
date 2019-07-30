function [Eft,Varft,theta] = testme_sexp_1d(x,y,xt,m)
%% testme_sexp_1d - Minimum working example for the sexp covariance function
%
% Syntax:
%   [Eft,Varft,theta] = testme_sexp_1d(x,y,xt,m)
%
% In:
%   x   - Training inputs
%   y   - Training targets
%   xt  - Test inputs
%   m   - Number of basis functions
%
% Out:
%   Eft   - Test target posterior mean
%   Varft - Test target posterior variance
%   theta - ML estimate of optimized hyperparameters
%   
% Description:
%   Minimum working example for the method in the paper [1].
%   This simplified example demonstrates how the method can be
%   implemented for the squared exponential covariance function in a
%   one-dimensional domain.
%
%   This code follows the presentation in the article, and is not optimized
%   for speed. For more exotic domains or other covariance functions, only
%   the spectral density and functions eigenval/eigenfun need to be
%   replaced.
%
% References:
%   [1] Arno Solin and Simo Sarkka (2019). Hilbert Space Methods for 
%       Reduced-Rank Gaussian Process Regression. Statistics and Computing.
%
% Copyright: 
%   (c) Arno Solin, 2013-2019
% 
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
% http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
%
%%

  % Ensure column vectors and center data
  x = x(:); xt = xt(:); y = y(:);
  mid = (min(x) + max(x))/2;
  x = x - mid; xt = xt - mid;  
  
  % Constants
  Lt = 4/3*max(abs(x));   % Boundary, x \in [-Lt,Lt]
  NN = (1:m)';            % Indices for eigenvalues
  w0 = log([.1 .1 .1]);   % Initial  hyperparameters
  
  % Define eigenfunction of the Laplacian in 1D
  eigenval = @(n) (n(:)'*pi/2/Lt).^2;
  eigenfun = @(n,x) Lt^(-1/2)*sin(kron(n(:)'*pi,(x(:)+Lt)/2/Lt));
  
  % The spectral density of the squared exponential covariance function
  S = @(w,lengthScale,magnSigma2) ...
            magnSigma2*sqrt(2*pi)*lengthScale*exp(-w.^2*lengthScale^2/2); 
  
  % Derivative w.r.t. lengthScale  
  dS{1} = @(w,lengthScale,magnSigma2) ...
            magnSigma2/lengthScale*((2*pi*lengthScale^2)^(1/2)* ...
            exp(-lengthScale^2*w.^2/2)).*(1-lengthScale^2*w.^2);  
  
  % Derivative w.r.t. magnSigma2  
  dS{2} = @(w,lengthScale,magnSigma2) ...
            S(w,lengthScale,magnSigma2)/magnSigma2;
        
  % Eigenfunctions
  Phit   = eigenfun(NN,xt);
  Phi    = eigenfun(NN,x); 
  PhiPhi = Phi'*Phi;        % O(nm^2)
  Phiy   = Phi'*y;
  lambda = eigenval(NN)';
    
  % Optimize hyperparameters
  fun = @(w) opthyperparams(w,y,lambda,Phiy,PhiPhi,S,dS);
  options = optimset('Display','iter','GradObj','on', ...
                     'TolX',1e-5,'TolFun',1e-5);
  
  % Optimize
  [w,~,~,output] = fminunc(fun,w0,options);
    
  % Extract hyperparameters
  lengthScale = exp(w(1)); % Length-scale parameter 
  magnSigma2  = exp(w(2)); % Magnitude parameter
  sigma2      = exp(w(3)); % Measurement noise variance
  
  % Solve GP with optimized hyperparameters and 
  % return predictive mean and variance 
  k = S(sqrt(lambda),lengthScale,magnSigma2);
  L = chol(PhiPhi + diag(sigma2./k),'lower'); 
  Eft = Phit*(L'\(L\Phiy));
  Varft = sigma2*sum((Phit/L').^2,2); 
  
  % Notice boundaries
  Eft(abs(xt) > Lt) = 0; Varft(abs(xt) > Lt) = 0;
  
  % Return optimized hyperparameters
  theta = exp(w);
  
end

function [e,eg] = opthyperparams(w,y,lambda,Phiy,PhiPhi,S,dS)

  % Extract parameters
  lengthScale = exp(w(1)); % Length scale parameter 
  magnSigma2  = exp(w(2)); % Magnitude parameter
  sigma2      = exp(w(3)); % Measurement noise variance
  
  % Evaluate the spectral density
  k = S(sqrt(lambda),lengthScale,magnSigma2);
  
  % Number of n=observations and m=basis functions
  n = numel(y);
  m = size(Phiy,1);
  
  % Calculate the Cholesky factor
  L = chol(PhiPhi + diag(sigma2./k),'lower');  % O(m^3)
  
  % Evaluate all parts
  v = L\Phiy; % Phiy = (Phi'*y);
  yiQy = (y'*y - v'*v)/sigma2;
  logdetQ = (n-m)*log(sigma2) + sum(log(k)) + 2*sum(log(diag(L)));
  
  % Return approx. negative log marginal likelihood
  e = .5*yiQy + .5*logdetQ + .5*n*log(2*pi);
  
  % Precalculate
  vv = L'\v;
  LLk = L'\(L\diag(1./k)); % O(m^3)
  
  % For the covariance function hyperparameters
  eg = zeros(size(w));
  for j=1:numel(w)-1
        
    % Evaluate the partial derivative
    dk = dS{j}(sqrt(lambda),lengthScale,magnSigma2);
        
    % Evaluate parts
    dlogdetQ = sum(dk./k) - sigma2*sum(diag(LLk).*dk./k);
    dyiQy = -vv'*diag(dk./k.^2)*vv;
        
    % The partial derivative
    eg(j) = .5*dlogdetQ + .5*dyiQy;
    
    % Account for the log-transformed values
    eg(j) = exp(w(j))*eg(j);
    
  end
  
  % Evaluate parts
  dlogdetQ = (n-m)/sigma2 + sum(diag(LLk));
  dyiQy    = vv'*diag(1./k)*vv/sigma2 - yiQy/sigma2;
  
  % For the measurement noise
  eg(end)  = .5*dlogdetQ + .5*dyiQy;
    
  % Account for the log-transformed values
  eg(end) = exp(w(end))*eg(end);
    
end
  