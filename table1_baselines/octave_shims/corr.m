function [r, p] = corr(x, y)
% Octave shim for MATLAB Statistics Toolbox corr(x, y) with Pearson p-values.
% r(i,j) = Pearson correlation of x(:,i) and y(:,j); p = two-sided t-test, n-2 df
% (identical to MATLAB's corr for 'type','Pearson').
  if nargin < 2, y = x; end
  n = size(x, 1);
  xc = x - mean(x, 1); yc = y - mean(y, 1);
  r = (xc' * yc) ./ (sqrt(sum(xc .^ 2, 1))' * sqrt(sum(yc .^ 2, 1)));
  r = max(min(r, 1), -1);
  if nargout > 1
    df = n - 2;
    t2 = r .^ 2 .* df ./ max(1 - r .^ 2, realmin);
    p = betainc(df ./ (df + t2), df / 2, 0.5);
    p(abs(r) >= 1) = 0;
  end
end
