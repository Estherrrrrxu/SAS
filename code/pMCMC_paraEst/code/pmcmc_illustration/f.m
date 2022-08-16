function xt1 = f(xt,t)
xt1 = 0.5*xt + 25*xt./(1+xt.^2) + 8*cos(1.2*t);