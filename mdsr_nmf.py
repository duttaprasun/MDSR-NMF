import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class An3mf_g:
    def __init__(self, nme, x, dc, factor, red_dim):
        print("An3mf_g init begins")
        self.nme = nme
        self.x = x
        self.dc = dc
        self.factor = factor
        self.r = red_dim
        self.m, self.n = self.x.shape
        print("m: ", self.m, "\tn: ", self.n, "\tr: ", self.r)
        ##### Xaviers initialization: np.random.randn(size_l, size_l-1)*np.sqrt(2 / (size_l + size_l-1)) #####
        self.v = np.random.randn(self.r, self.n)*np.sqrt(2 / (self.r + self.n))
        self.v = self.v.T
        self.y = np.zeros((self.m, self.r))
        self.b = np.zeros((self.m, self.r))
        ##### Xaviers initialization: np.random.randn(size_l, size_l-1)*np.sqrt(2 / (size_l + size_l-1)) #####
        self.w = np.random.randn(self.n, self.r)*np.sqrt(2 / (self.n + self.r))
        self.w = self.w.T
        self.z = np.zeros((self.m, self.n))
        self.x1 = np.zeros((self.m, self.n))
        self.I = np.identity(self.n)
        self.vw = np.zeros((self.n, self.n))
        self.vwi = np.zeros((self.n, self.n))
        self.lambdaa = 0.1
        self.eta_v = 0.1
        self.momentum = 0.9
        self.momentum_v = np.zeros(self.v.shape)
        self.momentum_w = np.zeros(self.w.shape)
        self.step = 0
        self.J_old = 1.0
        self.J = 1.1
        self.J1 = 0.0
        self.J2 = 0.0
        self.costarray = np.array([0, 0, 0, 0])
        print("An3mf_g init ends")
    
    def forward_propagation(self):
        print("An3mf_g forward propagation begins")
        self.y = np.matmul(self.x, self.v)
        self.b = General().sigmoid1(self.y)
        self.z = np.matmul(self.b, self.w)
        self.x1 = General().sigmoid1(self.z)
        print("An3mf_g forward propagation ends")
    
    def cost(self):
        print("An3mf_g cost begins")
        self.J1 = (1 / (2 * self.m * self.n)) * ((np.square(self.x - self.x1)).sum())
        self.J2 = (self.lambdaa / (2 * self.n * self.n)) * ((np.square(self.vwi)).sum())
        self.J = self.J1 + self.J2
        print("An3mf_g cost ends")
    
    def backward_propagation(self):
        print("An3mf_g backward propagation begins")
        w11 = (-1 / (self.m * self.n)) * np.matmul(self.b.T, ((self.x - self.x1) * self.x1 * (1 - self.x1))) 
        w12 = (self.lambdaa / (self.n * self.n)) * np.matmul(self.v.T, self.vwi)
        eta_w = np.divide(((self.m * self.n) * self.w), np.matmul(self.b.T, ((self.x1 * self.x1) + (self.x * self.x1 * self.x1))))
        w1 = self.w - eta_w * (w11 + w12)
        w1 = General().relu1(w1)
        v11 = (-1 / (self.m * self.n)) * np.matmul(self.x.T, (np.matmul(((self.x - self.x1) * self.x1 * (1 - self.x1)), self.w.T) * self.b * (1 - self.b)))
        v12 = ((self.lambdaa / (self.n * self.n)) * np.matmul(self.vwi, self.w.T))
        self.momentum_v = self.momentum * self.momentum_v - self.eta_v * (v11 + v12)
        v1 = self.v + self.momentum_v
        self.w = w1
        self.v = v1
        print("An3mf_g backward propagation ends")
    
    def model(self):
        print("An3mf_g model begins")
        while self.step < 1:#20
            if self.step != 0 :
                self.v = np.random.randn(self.r, self.n)*np.sqrt(2 / (self.r + self.n))
                self.v = self.v.T
                self.w = np.random.randn(self.n, self.r)*np.sqrt(2 / (self.n + self.r))
                self.w = self.w.T
                self.step = 0
                self.J_old = 1.0
                self.J = 1.1
                print("reinitialized")
            while self.step < 5:#self.J > 1 or abs(self.J_old - self.J) > 0.0000001:#
                self.step = self.step + 1
                self.J_old = self.J
                self.forward_propagation()
                self.vw = np.matmul(self.v, self.w)
                self.vwi = self.vw - self.I
                self.cost()
                self.backward_propagation()
                self.costarray = np.append(self.costarray, [self.step, self.J1, self.J2, self.J])
                print("An3mf_g step:", self.step, "J1:", np.around(self.J1, decimals = 10), "J2:", np.around(self.J2, decimals = 10), "J:", np.around(self.J, decimals = 10))
            fname1 = self.nme + "/" + self.nme + "_an3mfg_x_" + str(self.dc) + "_" + str(self.r) + "_" + str(self.factor) + ".txt"
            f1 = open(fname1, "w")
            np.savetxt(f1, self.x, fmt = '%.17f', delimiter = ',')
            f1.close()
            fname2 = self.nme + "/" + self.nme + "_an3mfg_v_" + str(self.dc) + "_" + str(self.r) + "_" + str(self.factor) + ".txt"
            f2 = open(fname2, "w")
            np.savetxt(f2, self.v, fmt = '%.17f', delimiter = ',')
            f2.close()
            fname3 = self.nme + "/" + self.nme + "_an3mfg_b_" + str(self.dc) + "_" + str(self.r) + "_" + str(self.factor) + ".txt"
            f3 = open(fname3, "w")
            np.savetxt(f3, self.b, fmt = '%.17f', delimiter = ',')
            f3.close()
            fname4= self.nme + "/" + self.nme + "_an3mfg_w_" + str(self.dc) + "_" + str(self.r) + "_" + str(self.factor) + ".txt"
            f4 = open(fname4, "w")
            np.savetxt(f4, self.w, fmt = '%.17f', delimiter = ',')
            f4.close()
            fname5 = self.nme + "/" + self.nme + "_an3mfg_x1_" + str(self.dc) + "_" + str(self.r) + "_" + str(self.factor) + ".txt"
            f5 = open(fname5, "w")
            np.savetxt(f5, self.x1, fmt = '%.17f', delimiter = ',')
            f5.close()
            costmatrix = self.costarray.reshape(int(self.costarray.size/4), 4)
            costmatrix = np.delete(costmatrix, (0), axis = 0)
            fname6 = self.nme + "/" + self.nme + "_an3mfg_costmatrix_" + str(self.dc) + "_" + str(self.r) + "_" + str(self.factor) + ".txt"
            f6 = open(fname6, "w")
            np.savetxt(f6, costmatrix, fmt = '%.17f', delimiter = ',')
            f6.close()
            plt.xlabel('iteration') 
            plt.ylabel('cost') 
            titl = self.nme + "_cost_vs_iteration"
            plt.title(titl) 
            plt.plot(costmatrix[:, 3:])
            imgname = self.nme + "_an3mfg_" + str(self.dc) + "_" + str(self.r) + "_" + str(self.factor) + ".png"
            plt.savefig(self.nme + "/" + imgname)
            plt.clf()
            fname7 = self.nme + "/" + self.nme + "_an3mfg_size_cost_" + str(self.dc) + "_" + str(self.r) + "_" + str(self.factor) + ".txt"
            f7 = open(fname7, "w")
            f7.write("%s, %s, %s, %s, %s, %s\n" % (self.m, self.n, self.step, self.J1, self.J2, self.J))
            f7.close()
        print("An3mf_g model ends")

class Dn3mf:
    def __init__(self, nme, x, dc, factor, red_dim):
        print("Dn3mf init begins")
        self.nme = nme
        self.x = x
        self.dc = dc
        self.factor = factor
        self.red_dim = red_dim
        print("Dn3mf init ends")
    
    def model(self):
        print("Dn3mf model begins")
        for itr in range(len(self.red_dim)):
            if (itr == 0):
                dc = self.dc
            obj = An3mf_g(self.nme, self.x, dc, self.factor[itr], self.red_dim[itr])
            obj.model()
            fn1 = self.nme + "/" + self.nme + "_an3mfg_b_" + str(dc) + "_" + str(self.red_dim[itr]) + "_" + str(self.factor[itr]) + ".txt"
            f1 = open(fn1, "r")
            self.x = np.genfromtxt(f1, delimiter = ',')
            dc = self.red_dim[itr]
        print("Dn3mf model ends")

class Dn3mf2:
    def __init__(self, nme, x, dc, factor, red_dim):
        print("Dn3mf2 init begins")
        self.nme = nme
        self.x = x
        self.dc = dc
        self.factor = factor
        self.red_dim = red_dim
        print(self.factor, self.red_dim)
        self.m, self.n = self.x.shape
        self.dct = {}
        obj = Dn3mf(self.nme, self.x, self.dc, self.factor, self.red_dim)
        obj.model()
        self.dct["x0"] = self.x
        self.l = 1
        for fac in self.factor:
            if (self.l == 1):
                dc = self.dc
            fn1 = self.nme + "/" + self.nme + "_an3mfg_v_" + str(dc) + "_" + str(self.red_dim[self.l - 1]) + "_" + str(fac) + ".txt"
            f1 = open(fn1, "r")
            self.dct["v{0}".format(self.l)] = np.genfromtxt(f1, delimiter = ',')
            f1.close()
            v_r, v_c = self.dct["v{0}".format(self.l)].shape
            self.dct["momentum_v{0}".format(self.l)] = np.zeros((v_r, v_c))
            self.dct["x{0}".format(self.l)] = np.zeros((self.m, v_c))
            dc = self.red_dim[self.l - 1]
            self.l += 1
        self.l = self.l - 1
        
        obj = An3mf_g(self.nme, self.x, self.dc, self.factor[-1], self.red_dim[-1])
        obj.model()
        
        fn2 = self.nme + "/" +self. nme + "_an3mfg_w_" + str(self.dc) + "_" + str(self.red_dim[self.l - 1]) + "_" + str(fac) + ".txt"
        f2 = open(fn2, "r")
        self.dct["w{0}".format(0)] = np.genfromtxt(f2, delimiter = ',')
        f2.close()
        w_r, w_c = self.dct["w{0}".format(0)].shape
        self.dct["momentum_w{0}".format(0)] = np.zeros((w_r, w_c))
        self.dct["x{0}hat".format(0)] = np.zeros((self.m, w_c))
        self.I = np.identity(self.n)
        self.vw = np.zeros((self.n, self.n))
        self.vwi = np.zeros((self.n, self.n))
        self.lambdaa = 0.1
        self.eta_w = 0.1
        self.eta_v = 0.1
        self.momentum = 0.9
        self.step = 0
        self.J_old = 1.0
        self.J = 1.1
        self.J1 = 0.0
        self.J2 = 0.0
        self.costarray = np.array([0, 0, 0, 0])
        print("Dn3mf2 init ends")
    
    def forward_propagation(self):
        print("Dn3mf2 forward propagation begins")
        for itr in range(self.l):
            self.dct["x{0}".format(itr+1)] = np.matmul(self.dct["x{0}".format(itr)], self.dct["v{0}".format(itr+1)])
            self.dct["x{0}".format(itr+1)] = General().sigmoid1(self.dct["x{0}".format(itr+1)])
        self.dct["x{0}hat".format(0)] = np.matmul(self.dct["x{0}".format(self.l)], self.dct["w{0}".format(0)])
        self.dct["x{0}hat".format(0)] = General().sigmoid1(self.dct["x{0}hat".format(0)])
        print("Dn3mf2 forward propagation ends")
    
    def cost(self):
        print("Dn3mf2 cost begins")
        self.J1 = (1 / (2 * self.m * self.n)) * ((np.square(self.dct["x{0}".format(0)] - self.dct["x{0}hat".format(0)])).sum())
        self.J2 = (self.lambdaa / (2 * self.n * self.n)) * ((np.square(self.vwi)).sum())
        self.J = self.J1 + self.J2
        print("Dn3mf2 cost ends")
    
    def bp_vw(self, v_s, v_e, w_flag):
        for itr in range(v_s, v_e+1):
            if (itr == v_s):
                v = self.dct["v{0}".format(itr)]
            else:
                v = np.matmul(v, self.dct["v{0}".format(itr)])
        if (w_flag == 0):
            bpvw = v
        else:
            bpvw = np.matmul(v, self.dct["w{0}".format(0)])
        return bpvw
    
    def backward_propagation(self):
        print("Dn3mf2 backward propagation begins")
        dct1 = {}
        q1 = 0
        theta = ((self.dct["x{0}".format(0)] - self.dct["x{0}hat".format(0)]) * self.dct["x{0}hat".format(0)] * (1 - self.dct["x{0}hat".format(0)]))
        w_a = (-1 / (self.m * self.n)) * (np.matmul(self.dct["x{0}".format(self.l)].T, theta))
        bpvw = self.bp_vw(1, self.l, 0)
        w_b = (self.lambdaa / (self.n * self.n)) * (np.matmul(bpvw.T, self.vwi))
        self.dct["momentum_w{0}".format(q1)] = self.momentum * self.dct["momentum_w{0}".format(q1)] - self.eta_w * (w_a + w_b)
        dct1["w{0}_u".format(q1)] = self.dct["w{0}".format(q1)] + self.dct["momentum_w{0}".format(q1)]
        dct1["w{0}_u".format(q1)] = self.dct["w{0}".format(q1)] - self.eta_w * (w_a + w_b)
        dct1["w{0}_u".format(q1)] = General().relu1(dct1["w{0}_u".format(q1)])
        for itr in range(self.l, 0, -1):
            q2 = itr
            if (q2 == self.l):
                theta = (np.matmul(theta, self.dct["w{0}".format(0)].T)) * self.dct["x{0}".format(self.l)] * (1 - self.dct["x{0}".format(self.l)])
                v_a = (-1 / (self.m * self.n)) * (np.matmul(self.dct["x{0}".format(q2-1)].T, theta))
                bpvw1 = self.bp_vw(1, q2-1, 0)
                bpvw2 = self.dct["momentum_w{0}".format(0)]
                v_b = (self.lambdaa / (self.n * self.n)) * np.matmul((np.matmul(bpvw1.T, self.vwi)), bpvw2.T)
            elif (q2 == 1):
                theta = (np.matmul(theta, self.dct["v{0}".format(q2+1)].T)) * self.dct["x{0}".format(q2)] * (1 - self.dct["x{0}".format(q2)])
                v_a = (-1 / (self.m * self.n)) * (np.matmul(self.dct["x{0}".format(q2-1)].T, theta))
                bpvw = self.bp_vw(q2+1, self.l, 1)
                v_b = (self.lambdaa / (self.n * self.n)) * (np.matmul(self.vwi, bpvw.T))
            else:
                theta = (np.matmul(theta, self.dct["v{0}".format(q2+1)].T)) * self.dct["x{0}".format(q2)] * (1 - self.dct["x{0}".format(q2)])
                v_a = (-1 / (self.m * self.n)) * (np.matmul(self.dct["x{0}".format(q2-1)].T, theta))
                bpvw1 = self.bp_vw(1, q2-1, 0)
                bpvw2 = self.bp_vw(q2+1, self.l, 1)
                v_b = (self.lambdaa / (self.n * self.n)) * np.matmul((np.matmul(bpvw1.T, self.vwi)), bpvw2.T)
            self.dct["momentum_v{0}".format(q2)] = self.momentum * self.dct["momentum_v{0}".format(q2)] - self.eta_v * (v_a + v_b)
            dct1["v{0}_u".format(q2)] = self.dct["v{0}".format(q2)] + self.dct["momentum_v{0}".format(q2)]
            dct1["v{0}_u".format(q2)] = self.dct["v{0}".format(q2)] - self.eta_v * (v_a + v_b)
        self.dct["w{0}".format(0)] = dct1["w{0}_u".format(0)]
        for itr in range(self.l):
            self.dct["v{0}".format(itr + 1)] = dct1["v{0}_u".format(itr + 1)]
        print("Dn3mf2 backward propagation ends")
    
    def model(self):
        print("Dn3mf2 model begins")
        while self.step < 1:#20
            while self.step < 5:#self.J > 1 or abs(self.J_old - self.J) > 0.0000001:#
                self.step = self.step + 1
                self.J_old = self.J
                self.forward_propagation()
                v = self.I
                for itr in range(self.l):
                    v = np.matmul(v, self.dct["v{0}".format(itr+1)])
                v_r, v_c = v.shape
                w = self.dct["w{0}".format(0)]
                self.vw = np.matmul(v, w)
                self.vwi = self.vw - self.I
                self.cost()
                self.backward_propagation()
                self.costarray = np.append(self.costarray, [self.step, self.J1, self.J2, self.J])
                print("Dn3mf2 step:", self.step, "J1:", np.around(self.J1, decimals = 10), "J2:", np.around(self.J2, decimals = 10), "J:", np.around(self.J, decimals = 10))
            fname1 = self.nme + "/" + self.nme + "_dn3mf2_b_" + str(self.factor[self.l - 1]) + ".txt"
            f1 = open(fname1, "w")
            np.savetxt(f1, self.dct["x{0}".format(self.l)], fmt = '%.17f', delimiter = ',')
            f1.close()
            costmatrix = self.costarray.reshape(int(self.costarray.size/4), 4)
            costmatrix = np.delete(costmatrix, (0), axis = 0)
            fname2 = self.nme + "/" + self.nme + "_dn3mf2_costmatrix_"+ str(self.factor[self.l - 1]) + ".txt"
            f2 = open(fname2, "w")
            np.savetxt(f2, costmatrix, fmt = '%.17f', delimiter = ',')
            f2.close()
            plt.xlabel('iteration') 
            plt.ylabel('cost') 
            titl = self.nme + "_cost_vs_iteration"
            plt.title(titl) 
            plt.plot(costmatrix[:, 3:])
            imgname = self.nme + "_dn3mf2_" + str(self.m) + "_" + str(self.n) + "_" + str(self.factor[self.l - 1]) + ".png"
            plt.savefig(self.nme+"/"+imgname)
            plt.clf()
            fname3 = self.nme + "/" + self.nme + "_dn3mf2_size_cost_" + str(self.factor[self.l - 1]) + ".txt"
            f3 = open(fname3, "w")
            f3.write("%s, %s, %s, %s, %s, %s\n" % (self.m, self.n, self.step, self.J1, self.J2, self.J))
            f3.close()
        print("Dn3mf2 model ends")

class General:
    def readData(self, choice):
        print("data choice", choice)
        if choice == 1 :
            nme = "GastrointestinalLesionsInRegularColonoscopy"
            fn = nme + "/" + "data.txt"
            f = open(fn, "r")
            d1 = np.genfromtxt(f, delimiter = ',')
            d2 = d1[1:, 2:701]
        elif choice == 2 :
            nme = "OnlineNewsPopularity"
            fn = nme + "/" + nme + ".csv"
            f = open(fn, "r")
            d1 = np.genfromtxt(f, delimiter = ',')
            d2 = d1[1:, 1:60]
        elif choice == 3 :
            nme = "ParkinsonsDiseaseClassification"
            fn = nme + "/" + "pd_speech_features.csv"
            f = open(fn, "r")
            d1 = np.genfromtxt(f, delimiter = ',')
            d2 = d1[2:, 1:754]
        elif choice == 4 :
            nme = "StudentPerformance"
            fn = nme + "/" + "student_mat.csv"
            f = open(fn, "r")
            d1 = np.genfromtxt(f, delimiter = ',')
            d2 = d1[1:, :32]
        elif choice == 5 :
            nme = "MovieLens"
            fn = nme + "/" + "data_labled.csv"
            df = pd.read_csv(fn)
            d1 = df
            d2 = df.iloc[:,3:-3].to_numpy()
        else :
            print("wrong cmd line i/p")
        if choice != 5 :
            f.close()
        print("original data shape:\t", d1.shape, "\nreduced data shape:\t", d2.shape)
        return (nme, d2)
    
    def z_score_normalize(self, matrx):
        r, c = matrx.shape
        mn = np.mean(matrx, axis = 0)
        sd = np.std(matrx, axis = 0)
        for j in range(c):
            if (sd[j] == 0):
                sd[j] = 0.01
            matrx[:, j] = (matrx[:, j] - mn[j]) / sd[j]
        return matrx
    
    def min_max_normalize(self, matrx):
        r, c = matrx.shape
        mini = np.amin(matrx, axis = 0)
        maxi = np.amax(matrx, axis = 0)
        new_min = 0.0
        new_max = 1.0
        for j in range(c):
            scale = maxi[j] - mini[j]
            matrx[:, j] = (((matrx[:, j] - mini[j]) / scale) * (new_max - new_min)) + new_min
        return matrx
    
    def col_double(self, d1):
        d2 = d1.copy()
        d3 = d1.copy()
        d2[d2 < 0] = 0
        d3[d3 > 0] = 0
        d3 = np.absolute(d3)
        d4 = np.append(d2, d3, axis = 1) #col append
        return d4
    
    def add_noise(self, matrx):
        r, c = matrx.shape
        mini = np.amin(matrx, axis = 0)
        maxi = np.amax(matrx, axis = 0)
        for j in range(c):
            if (maxi[j] == mini[j]):
                matrx[:, j] = matrx[:, j] + np.random.normal(0, .1, matrx[:, j].size)
        return matrx
    
    def preprocess(self, nme, d):
        d1 = self.z_score_normalize(d)
        d2 = self.col_double(d1)
        print("processed data shape:\t", d2.shape)
        x = d2[:, :]
        if (x.min() < 0):
            print("Input matrix elements can not be negative!!!")
        print("working data shape:\t", x.shape)
        fn = nme + "/" + nme + "_processed_data.txt"
        f = open(fn, "w")
        np.savetxt(f, x, fmt = '%.17f', delimiter = ',')
        f.close()
        return x
    
    def relu1(self, matrx):
        r, c = matrx.shape
        for i in range (r):
            for j in range (c):
                if (matrx[i][j] <= 0):
                    matrx[i][j] = 0.001
        return matrx
    
    def tanh1(self, matrx):
        r, c = matrx.shape
        for i in range (r):
            for j in range (c):
                matrx[i][j] = (np.exp(matrx[i][j]) - np.exp(-matrx[i][j])) / (np.exp(matrx[i][j]) + np.exp(-matrx[i][j]))
                if (matrx[i][j] == 0):
                    matrx[i][j] = 0.001
        return matrx
    
    def sigmoid1(self, matrx):
        r, c = matrx.shape
        for i in range (r):
            for j in range (c):
                matrx[i][j] = 1.0 / (1.0 + np.exp(-matrx[i][j]))
                if (matrx[i][j] == 0):
                    matrx[i][j] = 0.001
        return matrx

def main(data_choice, f):
    data_choice = int(data_choice)
    obj1 = General()
    nme, d = obj1.readData(data_choice)
    dr, dc = d.shape
    x = obj1.preprocess(nme, d)
    target_factor2 = f
    target_factor1 = 1.0 - (1.0-target_factor2)/2.0
    factor = np.array([target_factor1, target_factor2])
    red_dim = dc * factor
    red_dim = red_dim.astype(int)
    print("factor : ", factor, ", red_dim : ", red_dim)
    obj4 = Dn3mf2(nme, x, dc, factor, red_dim)
    obj4.model()

if __name__ == '__main__':
    data_choice = 5 # 1: GastrointestinalLesionsInRegularColonoscopy, 2: OnlineNewsPopularity, 3: ParkinsonsDiseaseClassification, 4: StudentPerformance, 5: MovieLens
    f = 0.25
    main(data_choice, f)