import numpy as np


class ParityLine:
    def __init__(self, MANAGER=""):
        self.MANAGER = hash(MANAGER)
        self.a = 7876 * 1e-4
        self.b = 8 * 1e-2
        self.weight_A = 8 * 1e-1
        self.weight_B = 2 * 1e-1
        self.sigma_a = 6275 * 1e-4
        self.sigma_b = 6127 * 1e-4
        self.sigma_c = 473 * 1e-4
        self.r_a = 5 * 1e-1
        self.r_b = 2 * 1e-1
        self.r_c = 8 * 1e-2

    def getMinReturn(self):
        return self.r_c

    def getMaxReturn(self):
        return max(self.r_a, self.r_b)

    def getMinRisk(self):
        risk, _, _, _ = self.convertReturn(self.getMinReturn())
        return risk

    def getMaxRisk(self):
        risk, _, _, _ = self.convertReturn(self.getMaxReturn())
        return risk

    def setSigma(self, sigma_a, sigma_b, sigma_c):
        self.sigma_a, self.sigma_b, self.sigma_c = sigma_a, sigma_b, sigma_c

    def setParityLineCoeff(self, a, b):
        self.a, self.b = a, b

    def setReturns(self, r_a, r_b, r_c):
        self.r_a, self.r_b, self.r_c = r_a, r_b, r_c

    def setWeightCoeff(self, weight_A, weight_B):
        self.weight_A, self.weight_B = weight_A, weight_B

    def convertRisk(self, risk):
        assert (risk >= self.getMinRisk()) and (risk <= self.getMaxRisk())
        _return = (self.a * risk) + self.b
        weights = self._calculateWeights(_return)
        return _return, weights

    def convertReturn(self, _return):
        assert (_return >= self.getMinReturn()) and (_return <= self.getMaxReturn())
        risk = (_return - self.b) / self.a
        weights = self._calculateWeights(_return)
        return risk, weights[0], weights[1], weights[2]

    def convertWeights(self, weight_alpha, weight_beta, weight_gamma):
        _return = (
            weight_alpha * self.r_a + weight_beta * self.r_b + weight_gamma * self.r_c
        )
        risk = (_return - self.b) / self.a
        return risk, _return

    def _calculateWeights(self, _return):
        _combinedReturn = float(self.weight_A * self.r_a + self.weight_B * self.r_b)
        _w_Beta_Ind = 1 if _return <= _combinedReturn else 0
        _g_weight = self.g_weight(_return, _combinedReturn, _w_Beta_Ind)
        _c_weight = self.c_weight(_return, _combinedReturn, _w_Beta_Ind)
        _g_trim = np.clip(_g_weight, 0, 1)
        _c_trim = np.clip(_c_weight, 0, 1)
        weight_beta = _c_trim * self.weight_B * _w_Beta_Ind
        weight_gamma = _g_trim
        weight_alpha = 1 - weight_beta - weight_gamma
        return weight_alpha, weight_beta, weight_gamma

    def g_weight(self, _return, _combinedReturn, _w_Beta_Ind):
        if _w_Beta_Ind == 1:
            if _combinedReturn >= _return:
                return (_combinedReturn - _return) / (_combinedReturn - self.r_c)
        else:
            if self.r_a >= _return:
                return (self.r_a - _return) / (self.r_a - self.r_c)
        return 0

    def c_weight(self, _return, _combinedReturn, _w_Beta_Ind):
        if _w_Beta_Ind != 0 and _return >= self.r_c:
            return (_return - self.r_c) / (_combinedReturn - self.r_c)
        return 0
