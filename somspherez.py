"""
.. Spherical Self-Organizing-Maps
.. 球型SOM
.. Hugo Li

"""
from __future__ import print_function
from builtins import zip
from builtins import range
from builtins import object

__author__ = 'Hugo Li'

import numpy
import copy
import sys, os, random
import warnings

warnings.simplefilter("ignore", RuntimeWarning)

def get_index(ix, iy, nx, ny):
    return iy * nx + ix


def get_pair(ii, nx, ny):
    iy = int(numpy.floor(ii / nx))
    ix = ii % nx
    return ix, iy


def get_ns(ix, iy, nx, ny, index=False):
    """
    Get neighbors for rectangular grid given its coordinates and size of grid
    给定矩形网格的坐标和网格大小，获取其邻居

    :param int ix: Coordinate in the x-axis
    :param int ix: x轴上的坐标

    :param int iy: Coordinate in the y-axis
    :param int iy: y轴上的坐标

    :param int nx: Number fo cells along the x-axis
    :param int nx: 沿x轴的单元格数

    :param int ny: Number fo cells along the y-axis
    :param int ny: 沿y轴的单元格数

    :param bool index: Return indexes in the map format
    :param bool index: 返回映射格式的索引

    :return: Array of indexes for direct neighbors
    :return: 直接邻居的索引数组
    """
    ns = []
    if ix - 1 >= 0: ns.append((ix - 1, iy))
    if iy - 1 >= 0: ns.append((ix, iy - 1))
    if ix + 1 < nx: ns.append((ix + 1, iy))
    if iy + 1 < ny: ns.append((ix, iy + 1))

    if ix - 1 >= 0 and iy - 1 >= 0: ns.append((ix - 1, iy - 1))
    if ix - 1 >= 0 and iy + 1 < ny: ns.append((ix - 1, iy + 1))
    if ix + 1 < nx and iy + 1 < ny: ns.append((ix + 1, iy + 1))
    if ix + 1 < nx and iy - 1 >= 0: ns.append((ix + 1, iy - 1))

    ns = numpy.array(ns)
    if not index:
        return ns
    if index:
        ins = []
        for i in range(len(ns)):
            ins.append(get_index(ns[i, 0], ns[i, 1], nx, ny))
        return numpy.array(ins)


def get_ns_hex(ix, iy, nx, ny, index=False):
    """
    Get neighbors for hexagonal grid given its coordinates
    and size of grid
    Same parameters as :func:`get_ns`
    """
    ns = []
    even = False
    if iy % 2 == 0: even = True
    if ix - 1 >= 0: ns.append((ix - 1, iy))
    if ix + 1 < nx: ns.append((ix + 1, iy))
    if iy - 1 >= 0: ns.append((ix, iy - 1))
    if iy + 1 < ny: ns.append((ix, iy + 1))
    if even and ix - 1 >= 0 and iy - 1 >= 0: ns.append((ix - 1, iy - 1))
    if even and ix - 1 >= 0 and iy + 1 < ny: ns.append((ix - 1, iy + 1))
    if not even and ix + 1 < nx and iy - 1 >= 0: ns.append((ix + 1, iy - 1))
    if not even and ix + 1 < nx and iy + 1 < ny: ns.append((ix + 1, iy + 1))
    ns = numpy.array(ns)
    if not index:
        return ns
    if index:
        ins = []
        for i in range(len(ns)):
            ins.append(get_index(ns[i, 0], ns[i, 1], nx, ny))
        return numpy.array(ins)
# 几何
def geometry(top, Ntop, periodic='no'):
    """
    Pre-compute distances between cells in a given topology and store it on a distLib array
    预先计算给定拓扑中单元之间的距离,并将其存储在distLib数组中

    :param str top: Topology ('grid','hex','sphere')
    :param str top:拓扑结构（'grid'、'hex'、'sphere')

    :param int Ntop: Size of map,  for grid Size=Ntop*Ntop,
        for hex Size=Ntop*(Ntop+1[2]) if Ntop is even[odd] and for sphere
        Size=12*Ntop*Ntop and top must be power of 2
    :param int Ntop: map的大小
        grid = Ntop * Ntop
        hex = Ntop *(Ntop+1[2]), 如果Ntop是偶数[奇数]
        sphere = 12 * Ntop * Ntop, Ntop必须是2的幂

    :param str periodic: Use periodic boundary conditions ('yes'/'no'), valid for 'hex' and 'grid' only
    :param str periodic: 使用周期性边界条件（“是”/“否”), 仅对“hex”和“grid”有效

    :return: 2D array with distances pre computed between cells and total number of units
    :return: 2D数组, 其中单元格之间的距离和单元总数已预先计算

    :rtype: 2D float array, int
    :rtype: 2D浮点数组, int
    """

    if top == 'sphere':
        try:
            import healpy as hpx
        except:
            print('Error: healpy module not found, use grid or hex topologies')
            sys.exit(0)
    if top == 'sphere':
        nside = Ntop
        npix = 12 * nside ** 2
        distLib = numpy.zeros((npix, npix))
        for i in range(npix):
            ai = hpx.pix2ang(nside, i)
            for j in range(i + 1, npix):
                aj = hpx.pix2ang(nside, j)
                distLib[i, j] = hpx.rotator.angdist(ai, aj)
                distLib[j, i] = distLib[i, j]
        distLib[numpy.where(numpy.isnan(distLib))] = numpy.pi
    if top == 'grid':
        nx = Ntop
        ny = Ntop
        npix = nx * ny
        mapxy = numpy.mgrid[0:1:complex(0, nx), 0:1:complex(0, ny)]
        mapxy = numpy.reshape(mapxy, (2, npix))
        bX = mapxy[1]
        bY = mapxy[0]
        dx = 1. / (nx - 1)
        dy = 1. / (ny - 1)
        distLib = numpy.zeros((npix, npix))
        if periodic == 'no':
            for i in range(npix):
                for j in range(i + 1, npix):
                    distLib[i, j] = numpy.sqrt((bX[i] - bX[j]) ** 2 + (bY[i] - bY[j]) ** 2)
                    distLib[j, i] = distLib[i, j]
        if periodic == 'yes':
            for i in range(npix):
                for j in range(i + 1, npix):
                    s0 = numpy.sqrt((bX[i] - bX[j]) ** 2 + (bY[i] - bY[j]) ** 2)
                    s1 = numpy.sqrt((bX[i] - (bX[j] + 1. + dx)) ** 2 + (bY[i] - bY[j]) ** 2)
                    s2 = numpy.sqrt((bX[i] - (bX[j] + 1. + dx)) ** 2 + (bY[i] - (bY[j] + 1. + dy)) ** 2)
                    s3 = numpy.sqrt((bX[i] - (bX[j] + 0.)) ** 2 + (bY[i] - (bY[j] + 1. + dy)) ** 2)
                    s4 = numpy.sqrt((bX[i] - (bX[j] - 1. - dx)) ** 2 + (bY[i] - (bY[j] + 1. + dy)) ** 2)
                    s5 = numpy.sqrt((bX[i] - (bX[j] - 1. - dx)) ** 2 + (bY[i] - (bY[j] + 0.)) ** 2)
                    s6 = numpy.sqrt((bX[i] - (bX[j] - 1. - dx)) ** 2 + (bY[i] - (bY[j] - 1. - dy)) ** 2)
                    s7 = numpy.sqrt((bX[i] - (bX[j] + 0.)) ** 2 + (bY[i] - (bY[j] - 1. - dy)) ** 2)
                    s8 = numpy.sqrt((bX[i] - (bX[j] + 1. + dx)) ** 2 + (bY[i] - (bY[j] - 1. - dy)) ** 2)
                    distLib[i, j] = numpy.min((s0, s1, s2, s3, s4, s5, s6, s7, s8))
                    distLib[j, i] = distLib[i, j]
    if top == 'hex':
        nx = Ntop
        ny = Ntop
        xL = numpy.arange(0, nx, 1.)
        dy = 0.8660254
        yL = numpy.arange(0, ny, dy)
        ny = len(yL)
        nx = len(xL)
        npix = nx * ny
        bX = numpy.zeros(nx * ny)
        bY = numpy.zeros(nx * ny)
        kk = 0
        last = ny * dy
        for jj in range(ny):
            for ii in range(nx):
                if jj % 2 == 0: off = 0.
                if jj % 2 == 1: off = 0.5
                bX[kk] = xL[ii] + off
                bY[kk] = yL[jj]
                kk += 1
        distLib = numpy.zeros((npix, npix))
        if periodic == 'no':
            for i in range(npix):
                for j in range(i + 1, npix):
                    distLib[i, j] = numpy.sqrt((bX[i] - bX[j]) ** 2 + (bY[i] - bY[j]) ** 2)
                    distLib[j, i] = distLib[i, j]
        if periodic == 'yes':
            for i in range(npix):
                for j in range(i + 1, npix):
                    s0 = numpy.sqrt((bX[i] - bX[j]) ** 2 + (bY[i] - bY[j]) ** 2)
                    s1 = numpy.sqrt((bX[i] - (bX[j] + nx)) ** 2 + (bY[i] - bY[j]) ** 2)
                    s2 = numpy.sqrt((bX[i] - (bX[j] + nx)) ** 2 + (bY[i] - (bY[j] + last)) ** 2)
                    s3 = numpy.sqrt((bX[i] - (bX[j] + 0)) ** 2 + (bY[i] - (bY[j] + last)) ** 2)
                    s4 = numpy.sqrt((bX[i] - (bX[j] - nx)) ** 2 + (bY[i] - (bY[j] + last)) ** 2)
                    s5 = numpy.sqrt((bX[i] - (bX[j] - nx)) ** 2 + (bY[i] - (bY[j] + 0)) ** 2)
                    s6 = numpy.sqrt((bX[i] - (bX[j] - nx)) ** 2 + (bY[i] - (bY[j] - last)) ** 2)
                    s7 = numpy.sqrt((bX[i] - (bX[j] + 0)) ** 2 + (bY[i] - (bY[j] - last)) ** 2)
                    s8 = numpy.sqrt((bX[i] - (bX[j] + nx)) ** 2 + (bY[i] - (bY[j] - last)) ** 2)
                    distLib[i, j] = numpy.min((s0, s1, s2, s3, s4, s5, s6, s7, s8))
                    distLib[j, i] = distLib[i, j]
    
    return distLib, npix


def is_power_2(value):
    """
    Check if passed value is a power of 2
    检查值是否为2的幂
    """
    return value!=0 and ((value & (value- 1)) == 0)


def get_alpha(t, alphas, alphae, NT):
    """
    Get value of alpha at a given time
    获取给定时间的alpha值
    """
    return alphas * numpy.power(alphae / alphas, float(t) / float(NT))


def get_sigma(t, sigma0, sigmaf, NT):
    """
    Get value of sigma at a given time
    在给定时间获得sigam值
    """
    return sigma0 * numpy.power(sigmaf / sigma0, float(t) / float(NT))


def h(bmu, mapD, sigma):
    """
    Neighborhood function which quantifies how much cells around the best matching one are modified
    邻域函数，用于量化最佳匹配单元周围的单元数量

    :param int bmu: best matching unit
    :param int bmu: 最佳匹配单位

    :param float mapD: array of distances computed with :func:`geometry`
    :param float mapD: 使用 func:`geometry计算的距离数组`
    """
    return numpy.exp(-(mapD[bmu] ** 2) / sigma ** 2)


class SelfMap(object):
    """
    Create a som class instance
    创建一个som类实例

    :param float X: Attributes array (all columns used)    
    :param float X: 属性数组 (使用的所有列)

    :param float Y: Attribute to be predicted (not really needed, can be zeros)
    :param float Y: 要预测的属性(实际上不需要，可以是零)

    :param str topology: Which 2D topology, 'grid', 'hex' or 'sphere'
    :param str topology: 哪个2D拓扑是 “grid”、“hex” 还是 “sphere”

    :param str som_type: Which updating scheme to use 'online' or 'batch'
    :param str som_type: 使用“online”或“batch”的更新方案

    :param int Ntop: Size of map,  for grid Size=Ntop*Ntop,
        for hex Size=Ntop*(Ntop+1[2]) if Ntop is even[odd] and for sphere
        Size=12*Ntop*Ntop and top must be power of 2
    :param int Ntop: map的大小
        grid = Ntop * Ntop
        hex = Ntop *(Ntop+1[2]), 如果Ntop是偶数[奇数]
        sphere = 12 * Ntop * Ntop, Ntop必须是2的幂

    :param  int iterations: Number of iteration the entire sample is processed
    :param  int iterations: 处理整个样本的迭代次数

    :param str periodic: Use periodic boundary conditions ('yes'/'no'), valid for 'hex' and 'grid' only
    :param str periodic: 使用周期性边界条件（“是”/“否”), 仅对“hex”和“grid”有效

    :param dict dict_dim: dictionary with attributes names
    :param dict dict_dim: 具有属性名称的字典

    :param float astar: Initial value of alpha
    :param float astar: alpha的初始值

    :param float aend: End value of alpha
    :param float aend: alpha的结束值

    :param str importance: Path to the file with importance ranking for attributes, default is none
    :param str importance: 属性重要性排序文件的路径，默认为无

    """

    def __init__(self, X, Y, topology='grid', som_type='online', Ntop=28, iterations=30, periodic='no', dict_dim='',
                 astart=0.8, aend=0.5, importance=None):
        self.np, self.nDim = numpy.shape(X)
        self.dict_dim = dict_dim
        self.X = X
        # self.SF90 = SF90
        self.Y = Y
        self.aps = astart
        self.ape = aend
        self.top = topology
        if topology=='sphere' and not is_power_2(Ntop):
            print('Error, Ntop must be power of 2')
            sys.exit(0)
        self.stype = som_type
        self.Ntop = Ntop
        self.nIter = iterations
        self.per = periodic
        self.distLib, self.npix = geometry(self.top, self.Ntop, periodic=self.per)
        if importance == None: importance = numpy.ones(self.nDim)
        self.importance = importance / numpy.sum(importance)

    def som_best_cell(self, inputs, return_vals=1):
        """
        Return the closest cell to the input object
        It can return more than one value if needed
        返回距离输入对象最近的单元格
        如果需要，它可以返回多个值
        """
        activations = numpy.sum(numpy.transpose([self.importance]) * (
            numpy.transpose(numpy.tile(inputs, (self.npix, 1))) - self.weights) ** 2, axis=0)
        if return_vals == 1:
            best = numpy.argmin(activations)
            return best, activations
        else:
            best_few = numpy.argsort(activations)
            return best_few[0:return_vals], activations


    def create_map(self, evol='no', inputs_weights='', random_order=True):
        """
        This functions actually create the maps, it uses random values to initialize the weights
        这个函数实际上创建maps, 它使用随机值来初始化权重

        inputs_weights : 指定值初始化权重
        """

        if inputs_weights == '':
            self.weights = (numpy.random.rand(self.nDim, self.npix)) + self.X[0][0]
        else:
            self.weights = inputs_weights
        self.NT = self.nIter * self.np
        if self.stype == 'online':
            tt = 0
            sigma0 = self.distLib.max()
            sigma_single = numpy.min(self.distLib[numpy.where(self.distLib > 0.)])
            for it in range(self.nIter):
                #get alpha, sigma
                alpha = get_alpha(tt, self.aps, self.ape, self.NT)
                sigma = get_sigma(tt, sigma0, sigma_single, self.NT)
                if random_order:
                    index_random = random.sample(range(self.np), self.np)
                else:
                    index_random = numpy.arange(self.np)
                for i in range(self.np):
                    tt += 1
                    inputs = self.X[index_random[i]]
                    best, activation = self.som_best_cell(inputs)
                    self.weights += alpha * h(best, self.distLib, sigma) * numpy.transpose(
                        (inputs - numpy.transpose(self.weights)))
                if evol == 'yes':
                    self.evaluate_map()
                    self.save_map(itn=it)
        if self.stype == 'batch':
            tt = 0
            sigma0 = self.distLib.max()
            sigma_single = numpy.min(self.distLib[numpy.where(self.distLib > 0.)])
            for it in range(self.nIter):
                #get alpha, sigma
                sigma = get_sigma(tt, sigma0, sigma_single, self.NT)
                accum_w = numpy.zeros((self.nDim, self.npix))
                accum_n = numpy.zeros(self.npix)
                for i in range(self.np):
                    tt += 1
                    inputs = self.X[i]
                    best, activation = self.som_best_cell(inputs)
                    for kk in range(self.nDim):
                        accum_w[kk, :] += h(best, self.distLib, sigma) * inputs[kk]
                    accum_n += h(best, self.distLib, sigma)
                for kk in range(self.nDim):
                    self.weights[kk] = accum_w[kk] / accum_n

                if evol == 'yes':
                    self.evaluate_map()
                    self.save_map(itn=it)

    def evaluate_map(self, inputX='', inputY=''):
        """
        This functions evaluates the map created using the input Y or a new Y (array of labeled attributes)
        It uses the X array passed or new data X as well, the map doesn't change
        此函数用于计算使用输入Y或新Y(标记属性数组)创建的map
        使用传递的X数组或新数据X, map不会发生改变

        :param float inputX: Use this if another set of values for X is wanted using
            the weigths already computed
        :param float inputX: 如果需要使用已计算出的权重为X设置另一组值, 请使用此选项
        :param float inputY: One  dimensional array of the values to be assigned to each cell in the map
            based on the in-memory X passed
        :param float inputY: 基于内存中传递的X, 将值分配给映射中每个单元格的一维数组
        """
        self.yvals = {}
        self.ivals = {}
        if inputX == '':
            inX = self.X
        else:
            inX = inputX
        if inputY == '':
            inY = self.Y
        else:
            inY = inputY
        for i in range(len(inX)):
            inputs = inX[i]
            best, activation = self.som_best_cell(inputs)
            if best not in self.yvals: self.yvals[best] = []
            self.yvals[best].append(inY[i])
            if best not in self.ivals: self.ivals[best] = []
            self.ivals[best].append(i)

    def get_vals(self, line):
        """
        Get the predictions  given a line search, where the line is a vector of attributes per individual object fot the 10 closest cells.
        通过线搜索获得预测, "line"是10个最近单元中每个对象的属性向量。
        :param float line: input data to look in the tree
        :param float line: 输入要在树中查找的数据

        :return: array with the cell content
        :return: 包含单元格内容的数组
        """
        best, act = self.som_best_cell(line, return_vals=10)
        for ib in range(10):
            if best[ib] in self.yvals: return self.yvals[best[ib]]
        return numpy.array([-1.])

    def get_best(self, line):
        """
        Get the predictions  given a line search, where the line is a vector of attributes per individual object for THE best cell
        通过线搜索获得预测，"line" 是每个对象的最佳单元的属性向量

        :param float line: input data to look in the tree
        :param float line: 输入要在树中查找的数据

        :return: array with the cell content
        :return: 包含单元格内容的数组
        """
        best, act = self.som_best_cell(line, return_vals=10)
        return best[0]

    def save_map(self, itn=-1, fileout='SOM', path=''):
        """
        Saves the map

        :param int itn: Number of map to be included on path, use -1 to ignore this number
        要包含在路径上的地图数，请使用-1忽略此数字
        :param str fileout: Name of output file
        :param str path: path for the output file
        """
        if path == '':
            path = os.getcwd() + '/'
        if not os.path.exists(path): os.system('mkdir -p ' + path)
        if itn >= 0:
            ff = '_%04d' % itn
            fileout += ff
        numpy.save(path + fileout, self)

    def save_map_dict(self, path='', fileout='SOM', itn=-1):
        """
        Saves the map in dictionary format
        以字典格式保存地图

        :param int itn: Number of map to be included on path, use -1 to ignore this number
        要包含在路径上的地图数，请使用-1忽略此数字
        :param str fileout: Name of output file
        :param str path: path for the output file
        """
        SOM = {}
        SOM['W'] = self.weights
        SOM['yvals'] = self.yvals
        SOM['ivals'] = self.ivals
        SOM['topology'] = self.top
        SOM['Ntop'] = self.Ntop
        SOM['npix'] = self.npix
        if path == '':
            path = os.getcwd() + '/'
        if not os.path.exists(path): os.system('mkdir -p ' + path)
        if itn > 0:
            ff = '_%04d' % itn
            fileout += ff
        numpy.save(path + fileout, SOM)

    def plot_map(self, min_m=-100, max_m=-100, colbar='yes'):
        """
        Plots the map after evaluating, the cells are colored with the mean value inside each
        one of them
        在计算后绘制地图，用每个单元格内的平均值对单元格进行着色

        :param float min_m: Lower limit for coloring the cells, -100 uses min value
        单元格着色下限，-100使用最小值
        :param float max_m: Upper limit for coloring the cells, -100 uses max value
        单元格着色的上限，-100使用最大值
        :param str colbar: Include a colorbar ('yes','no')
        """

        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import matplotlib.cm as cm
        from matplotlib import collections, transforms
        from matplotlib.colors import colorConverter
        
        if self.top == 'sphere': import healpy as H

        if self.top == 'grid':
            M = numpy.zeros(self.npix) - 20.
            for i in range(self.npix):
                if i in self.yvals:
                    M[i] = numpy.mean(self.yvals[i])
            M2 = numpy.reshape(M, (self.Ntop, self.Ntop))
            plt.figure(figsize=(8, 8), dpi=100)
            if min_m == -100: min_m = M2[numpy.where(M2 > -10)].min()
            if max_m == -100: max_m = M2.max()
            # SM2 = plt.imshow(M2, origin='center', interpolation='nearest', cmap=cm.jet, vmin=min_m, vmax=max_m)
            SM2 = plt.imshow(M2, origin='lower', interpolation='nearest', cmap=cm.jet, vmin=min_m, vmax=max_m)
            SM2.cmap.set_under("grey")
            if colbar == 'yes': plt.colorbar()
            plt.axis('off')
        if self.top == 'hex':
            nx = self.Ntop
            ny = self.Ntop
            xL = numpy.arange(0, nx, 1.)
            dy = 0.8660254
            yL = numpy.arange(0, ny, dy)
            ny = len(yL)
            nx = len(xL)
            npix = nx * ny
            bX = numpy.zeros(nx * ny)
            bY = numpy.zeros(nx * ny)
            kk = 0
            for jj in range(ny):
                for ii in range(nx):
                    if jj % 2 == 0: off = 0.
                    if jj % 2 == 1: off = 0.5
                    bX[kk] = xL[ii] + off
                    bY[kk] = yL[jj]
                    kk += 1
            xyo = list(zip(bX, bY))
            sizes_2 = numpy.zeros(nx * ny) + ((8. * 0.78 / (self.Ntop + 0.5)) / 2. * 72.) ** 2 * 4. * numpy.pi / 3.
            M = numpy.zeros(npix) - 20.
            fcolors = [plt.cm.Spectral_r(x) for x in numpy.random.rand(nx * ny)]
            for i in range(npix):
                if i in self.yvals:
                    M[i] = numpy.mean(self.yvals[i])
            if max_m == -100: max_m = M.max()
            if min_m == -100: min_m = M[numpy.where(M > -10)].min()
            M = M - min_m
            M = M / (max_m - min_m)
            for i in range(npix):
                if M[i] <= 0:
                    fcolors[i] = plt.cm.Greys(.5)
                else:
                    fcolors[i] = plt.cm.jet(M[i])
            figy = ((8. * 0.78 / (self.Ntop + 0.5) / 2.) * (3. * ny + 1) / numpy.sqrt(3)) / 0.78
            fig3 = plt.figure(figsize=(8, figy), dpi=100)
            #fig3.subplots_adjust(left=0,right=1.,top=1.,bottom=0.)
            a = fig3.add_subplot(1, 1, 1)
            col = collections.RegularPolyCollection(6, sizes=sizes_2, offsets=xyo, transOffset=a.transData)
            col.set_color(fcolors)
            a.add_collection(col, autolim=True)
            a.set_xlim(-0.5, nx)
            a.set_ylim(-1, nx + 0.5)
            plt.axis('off')
            if colbar == 'yes':
                figbar = plt.figure(figsize=(8, 1.), dpi=100)
                ax1 = figbar.add_axes([0.05, 0.8, 0.9, 0.15])
                cmap = cm.jet
                norm = mpl.colors.Normalize(vmin=min_m, vmax=max_m)
                cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='horizontal')
                cb1.set_label('')
        if self.top == 'sphere':
            M = numpy.zeros(self.npix) + H.UNSEEN
            for i in range(self.npix):
                if i in self.yvals:
                    M[i] = numpy.mean(self.yvals[i])
            plt.figure(10, figsize=(8, 8), dpi=100)
            if min_m == -100: min_m = M[numpy.where(M > -10)].min()
            if max_m == -100: max_m = M.max()
            if colbar == 'yes': H.mollview(M, fig=10, title="", min=min_m, max=max_m, cbar=True)
            if colbar == 'no': H.mollview(M, fig=10, title="", min=min_m, max=max_m, cbar=False)
        plt.show()
