%% Test the example implementation
%
% Example of GP regression in 1D with the squared exponential
% covariance function, and hyperparameter ML optimization. The edge
% effects that are encountered in this particular case (Dirichlet BCs)
% are here dealt with by extending the boundary away from the training
% inputs.
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
  
%% The SINC function

  % Warnings may be thrown, as they are not dealt with
  warning off 

  % Sinc (simulate 2^15 training targets with noise)
  x = 3*pi*rand(1,32192);
  y = sinc(x) + .5*randn(size(x));

  % Define test points
  xt = linspace(-1,10,128);

  % Perform GP regression (64 = number of eigenfunctions)
  [Eft,Varft,theta] = testme_sexp_1d(x,y,xt,64);
  
  % Show
  figure(2); clf
  plot(x,y,'+',xt,Eft,'-k', ...
       xt,Eft+1.96*sqrt(Varft),'--k', ...
       xt,Eft-1.96*sqrt(Varft),'--k', ...
       xt,Eft+1.96*sqrt(Varft+theta(3)),'--r', ...
       xt,Eft-1.96*sqrt(Varft+theta(3)),'--r')
  xlabel('Input, x'); ylabel('Output, y'); axis tight

