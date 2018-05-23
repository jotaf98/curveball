
addpath ..

x_dim = 6;  % input size
h_dim = 6;  % hidden layer size
y_dim = 10;  % output size
num_samples = 8;

% 2-layers linear network model, large initial magnitudes
x = Input();
w1 = Param('value', randn(h_dim, x_dim, 'single'));
b1 = Param('value', randn(h_dim, 1, 'single'));
w2 = Param('value', randn(y_dim, h_dim, 'single'));
b2 = Param('value', randn(y_dim, 1, 'single'));

x2 = w1 * x + b1;
prediction = w2 * x2 + b2;

prediction.workspaceNames();
net = Net(prediction);

% assign input value (will be reused by all forward/backward calls)
x_value = rand(x_dim, num_samples, 'single');
net.setValue(x, x_value);


% create full matrix
H = zeros(y_dim * num_samples, 'single');
e = zeros(y_dim, num_samples, 'single');

for i = 1 : numel(e)
  e(i) = 1;  % one-hot vector / matrix
  Hcol = JJt(net, e) ;  % compute product
  H(:,i) = Hcol(:);  % reshape to vector and store it
  e(i) = 0;  % for next iter
end


figure(10); vl_tshow(H); title('H');

figure(11); vl_tshow(H - H'); title('Symmetric difference');


% computes product of J*J' and arbitrary vector y (shaped as net's output)
function y2 = JJt(net, y)
  % compute J' * y
  forward(net);
  p = backward(net, y);

  % compute J * J' * y
  y2 = fmad(net, p);
end

