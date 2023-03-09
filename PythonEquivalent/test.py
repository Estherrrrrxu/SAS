# %%
ns = 100
ps = 100
cl = 100
mean = 0.9
sd = 0.1
theta_record1 = pd.read_csv(f"theta_{ns}_{ps}_{cl}_{mean}_{sd}.csv",header = None)
plt.figure()
plt.subplot(1,3,1)
plt.hist(theta_record1,bins = 20,label = "theta_distribution")
x = np.linspace(ss.norm(mean,sd).ppf(0.01),ss.norm(mean,sd).ppf(0.99), 100)
plt.plot(x,ss.norm(mean,sd).pdf(x),label = "prior")
plt.xlabel(r"$\theta$")
plt.ylim([0,15])
plt.legend(frameon = False)
plt.title(r"$\theta \sim N(0.9,0.1)$")
plt.plot([0.9,0.9],[0,25],label = "prior mean")
plt.subplot(1,3,2)
sd = 0.2
plt.title(r"$\theta \sim N(0.9,0.2)$")
theta_record2 = pd.read_csv(f"theta_{ns}_{ps}_{cl}_{mean}_{sd}.csv",header = None)
plt.hist(theta_record2,bins = 20,label = "theta_distribution")
x = np.linspace(ss.norm(mean,sd).ppf(0.01),ss.norm(mean,sd).ppf(0.99), 100)
plt.plot(x,ss.norm(mean,sd).pdf(x),label = "prior")
plt.xlabel(r"$\theta$")
plt.plot([0.9,0.9],[0,25],label = "prior mean")
plt.ylim([0,15])
# plt.legend(frameon = False)
plt.subplot(1,3,3)
plt.title(r"$\theta \sim N(0.9,0.3)$")
sd = 0.3
theta_record3 = pd.read_csv(f"theta_{ns}_{ps}_{cl}_{mean}_{sd}.csv",header = None)
plt.hist(theta_record3,bins = 20,label = "theta_distribution")
x = np.linspace(ss.norm(mean,sd).ppf(0.01),ss.norm(mean,sd).ppf(0.99), 100)
plt.plot(x,ss.norm(mean,sd).pdf(x),label = "prior")
plt.plot([0.9,0.9],[0,25],label = "prior mean")
plt.xlabel(r"$\theta$")
plt.ylim([0,15])
# plt.legend(frameon = False)
# %%
ns = 100
ps = 100
cl = 100
mean = 0.8
sd = 0.2
theta_record1 = pd.read_csv(f"theta_{ns}_{ps}_{cl}_{mean}_{sd}.csv",header = None)
plt.figure()
plt.subplot(1,3,1)
plt.hist(theta_record1,bins = 20,label = "theta_distribution")
x = np.linspace(ss.norm(mean,sd).ppf(0.01),ss.norm(mean,sd).ppf(0.99), 100)
plt.plot(x,ss.norm(mean,sd).pdf(x),label = "prior")
plt.xlabel(r"$\theta$")
plt.ylim([0,15])
plt.legend(frameon = False)
plt.title(r"$\theta \sim N(0.8,0.2)$")
plt.plot([0.8,0.8],[0,25],label = "prior mean")
plt.subplot(1,3,2)
mean = 0.9
plt.title(r"$\theta \sim N(0.9,0.2)$")
theta_record2 = pd.read_csv(f"theta_{ns}_{ps}_{cl}_{mean}_{sd}.csv",header = None)
plt.hist(theta_record2,bins = 20,label = "theta_distribution")
x = np.linspace(ss.norm(mean,sd).ppf(0.01),ss.norm(mean,sd).ppf(0.99), 100)
plt.plot(x,ss.norm(mean,sd).pdf(x),label = "prior")
plt.xlabel(r"$\theta$")
plt.plot([0.9,0.9],[0,25],label = "prior mean")
plt.ylim([0,15])
# plt.legend(frameon = False)
plt.subplot(1,3,3)
plt.title(r"$\theta \sim N(1.0,0.2)$")
mean = 1.0
theta_record3 = pd.read_csv(f"theta_{ns}_{ps}_{cl}_{mean}_{sd}.csv",header = None)
plt.hist(theta_record3,bins = 20,label = "theta_distribution")
x = np.linspace(ss.norm(mean,sd).ppf(0.01),ss.norm(mean,sd).ppf(0.99), 100)
plt.plot(x,ss.norm(mean,sd).pdf(x),label = "prior")
plt.plot([1.0,1.0],[0,25],label = "prior mean")
plt.xlabel(r"$\theta$")
plt.ylim([0,15])
# plt.legend(frameon = False)

# %%
ns = 50
ps = 100
cl = 100
mean = 0.9
sd = 0.2
theta_record1 = pd.read_csv(f"theta_{ns}_{ps}_{cl}_{mean}_{sd}.csv",header = None)
plt.figure()
plt.subplot(1,3,1)
plt.hist(theta_record1,bins = 20,label = "theta_distribution")
x = np.linspace(ss.norm(mean,sd).ppf(0.01),ss.norm(mean,sd).ppf(0.99), 100)
plt.plot(x,ss.norm(mean,sd).pdf(x),label = "prior")
plt.xlabel(r"$\theta$")
plt.ylim([0,15])
plt.legend(frameon = False)
plt.title(f"particle N = {ns}")
plt.plot([0.9,0.9],[0,25],label = "prior mean")
plt.subplot(1,3,2)
ns = 100
plt.title(f"particle N = {ns}")
theta_record2 = pd.read_csv(f"theta_{ns}_{ps}_{cl}_{mean}_{sd}.csv",header = None)
plt.hist(theta_record2,bins = 20,label = "theta_distribution")
x = np.linspace(ss.norm(mean,sd).ppf(0.01),ss.norm(mean,sd).ppf(0.99), 100)
plt.plot(x,ss.norm(mean,sd).pdf(x),label = "prior")
plt.xlabel(r"$\theta$")
plt.plot([0.9,0.9],[0,25],label = "prior mean")
plt.ylim([0,15])
# plt.legend(frameon = False)
plt.subplot(1,3,3)
ns = 200
plt.title(f"particle N = {ns}")
theta_record3 = pd.read_csv(f"theta_{ns}_{ps}_{cl}_{mean}_{sd}.csv",header = None)
plt.hist(theta_record3,bins = 20,label = "theta_distribution")
x = np.linspace(ss.norm(mean,sd).ppf(0.01),ss.norm(mean,sd).ppf(0.99), 100)
plt.plot(x,ss.norm(mean,sd).pdf(x),label = "prior")
plt.plot([0.9,0.9],[0,25],label = "prior mean")
plt.xlabel(r"$\theta$")
plt.ylim([0,15])
# plt.legend(frameon = False)
# %%
ns = 100
ps = 50
cl = 100
mean = 0.9
sd = 0.2
theta_record1 = pd.read_csv(f"theta_{ns}_{ps}_{cl}_{mean}_{sd}.csv",header = None)
plt.figure()
plt.subplot(1,3,1)
plt.hist(theta_record1,bins = 20,label = "theta_distribution")
x = np.linspace(ss.norm(mean,sd).ppf(0.01),ss.norm(mean,sd).ppf(0.99), 100)
plt.plot(x,ss.norm(mean,sd).pdf(x),label = "prior")
plt.xlabel(r"$\theta$")
plt.ylim([0,15])
plt.legend(frameon = False)
plt.title(f"params D = {ps}")
plt.plot([0.9,0.9],[0,25],label = "prior mean")
plt.subplot(1,3,2)
ps = 100
plt.title(f"params D = {ps}")
theta_record2 = pd.read_csv(f"theta_{ns}_{ps}_{cl}_{mean}_{sd}.csv",header = None)
plt.hist(theta_record2,bins = 20,label = "theta_distribution")
x = np.linspace(ss.norm(mean,sd).ppf(0.01),ss.norm(mean,sd).ppf(0.99), 100)
plt.plot(x,ss.norm(mean,sd).pdf(x),label = "prior")
plt.xlabel(r"$\theta$")
plt.plot([0.9,0.9],[0,25],label = "prior mean")
plt.ylim([0,15])
# plt.legend(frameon = False)
plt.subplot(1,3,3)
ps = 200
plt.title(f"params D = {ps}")
theta_record3 = pd.read_csv(f"theta_{ns}_{ps}_{cl}_{mean}_{sd}.csv",header = None)
plt.hist(theta_record3,bins = 20,label = "theta_distribution")
x = np.linspace(ss.norm(mean,sd).ppf(0.01),ss.norm(mean,sd).ppf(0.99), 100)
plt.plot(x,ss.norm(mean,sd).pdf(x),label = "prior")
plt.plot([0.9,0.9],[0,25],label = "prior mean")
plt.xlabel(r"$\theta$")
plt.ylim([0,15])
# plt.legend(frameon = False)
# %%
ns = 100
ps = 100
cl = 50
mean = 0.9
sd = 0.2
theta_record1 = pd.read_csv(f"theta_{ns}_{ps}_{cl}_{mean}_{sd}.csv",header = None)
plt.figure()
plt.subplot(1,3,1)
plt.hist(theta_record1,bins = 20,label = "theta_distribution")
x = np.linspace(ss.norm(mean,sd).ppf(0.01),ss.norm(mean,sd).ppf(0.99), 100)
plt.plot(x,ss.norm(mean,sd).pdf(x),label = "prior")
plt.xlabel(r"$\theta$")
plt.ylim([0,15])
plt.legend(frameon = False)
plt.title(f"Chain length L = {cl}")
plt.plot([0.9,0.9],[0,25],label = "prior mean")
plt.subplot(1,3,2)
cl = 100
plt.title(f"Chain length L = {cl}")
theta_record2 = pd.read_csv(f"theta_{ns}_{ps}_{cl}_{mean}_{sd}.csv",header = None)
plt.hist(theta_record2,bins = 20,label = "theta_distribution")
x = np.linspace(ss.norm(mean,sd).ppf(0.01),ss.norm(mean,sd).ppf(0.99), 100)
plt.plot(x,ss.norm(mean,sd).pdf(x),label = "prior")
plt.xlabel(r"$\theta$")
plt.plot([0.9,0.9],[0,25],label = "prior mean")
plt.ylim([0,15])
# plt.legend(frameon = False)
plt.subplot(1,3,3)
ps = 200
cl = 200
plt.title(f"Chain length L = {cl}")
theta_record3 = pd.read_csv(f"theta_{ns}_{ps}_{cl}_{mean}_{sd}.csv",header = None)
plt.hist(theta_record3,bins = 20,label = "theta_distribution")
x = np.linspace(ss.norm(mean,sd).ppf(0.01),ss.norm(mean,sd).ppf(0.99), 100)
plt.plot(x,ss.norm(mean,sd).pdf(x),label = "prior")
plt.plot([0.9,0.9],[0,25],label = "prior mean")
plt.xlabel(r"$\theta$")
plt.ylim([0,15])
# plt.legend(frameon = False)
# %%
