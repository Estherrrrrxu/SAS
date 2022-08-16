function [x,y,q,r] = generate_data(T,q,r)
x = zeros(1, T);
y = zeros(1, T);
x(1) = 0; % Initial condition

for(t = 1:T)
    if(t < T)
        x(t+1) = f(x(t),t) + sqrt(q)*randn(1);
    end
    y(t) = h(x(t)) + sqrt(r)*randn(1);
end


