# a6.py
# Imani Chilongani, 
# 11/16/2016
"""K-means clustering algorithm to show visualization of data in csv files"""

import math
import random
import numpy


def is_point(thelist):
    """Return: True if thelist is a list of int or float """
    if (type(thelist) != list):
        return False
    
    # All float
    okay = True
    for x in thelist:
        if (not type(x) in [int,float]):
            okay = False
    
    return okay


def is_point_list(thelist):
    """Return: True if thelist is a 2D list of int or float, and every element
    of thelist has the same length """
    if thelist==None:
        return True

    if (type(thelist) != list):
        return False

    #All float
    okay = True
    for i in range(len(thelist)):
        point = is_point(thelist[i])
        if point==False:
            okay = False

    #All elements have same length
    length = len(thelist[0])
    for i in range(len(thelist)):
        if length!=len(thelist[i]):
            okay = False

    return okay


def is_valid_dim(d,contents):
    """Return: True if d is int and d==len(contents) or contents==None"""
    if type(d)!= int or d<=0:
        return False
    if contents==None:
        return True
    if d!=len(contents[0]):
        return False
    return True


def is_index(i,size):
    """Return: True if i is int and i<size """
    if type(i)!=int or i>=size:
        return False
    return True


def is_valid_seed_inds(seed_inds,length):
    """Return:True if seed_inds is a list and contains valid indices into ds"""
    if type(seed_inds)!=list:
        return False
    for i in seed_inds:
        if i>=length:
            return False
    return True





# CLASSES
class Dataset(object):
    """Instance is a dataset for k-means clustering.
    
    The data is stored as a list of list of numbers
    (ints or floats).  Each component list is a data point.
    
    Instance Attributes:
        _dimension: the point dimension for this dataset
            [int > 0. Value never changes after initialization]
        
        _contents: the dataset contents
            [a 2D list of numbers (float or int), possibly empty]:
    
    ADDITIONAL INVARIANT:
        The number of columns in _contents is equal to _dimension.  
        That is, for every item _contents[i] in the list _contents, 
        len(_contents[i]) == dimension.
    """
    
    def __init__(self, dim, contents=None):
        """Initializer: Creates a dataset for the given point dimension.
        
        Parameter dim: the initial value for attribute _dimension.  
        Precondition: dim is an int > 0.
        
        Parameter contents: the initial value for attribute _contents (optional). 
        Precondition: contents is either None or is a 2D list of numbers (int or float). 
        If contents is not None, then contents if not empty and the number of columns of 
        contents is equal to dim.
        """
        assert is_point_list(contents)
        assert is_valid_dim(dim,contents)

        self._dimension = dim
        self._contents =[]
        if contents!=None:
            for i in range(len(contents)):
                self._contents.append(contents[i][:])
    
    def getDimension(self):
        """Returns: The point dimension of this data set.
        """
        return self._dimension
    
    def getSize(self):
        """Returns: the number of elements in this data set.
        """
        return len(self._contents)
    
    def getContents(self):
        """Returns: The contents of this data set as a list.
        
        This method returns the attribute _contents directly.  Any changes made to this 
        list will modify the data set.  If you want to access the data set, but want to 
        protect yourself from modifying the data, use getPoint() instead.
        """
        return self._contents
    
    def getPoint(self, i):
        """Returns: A COPY of the point at index i in this data set.       
        
        Parameter i: the index position of the point
        Precondition: i is an int that refers to a valid position in 0..getSize()-1
        """
        # IMPLEMENT ME
        assert is_index(i,self.getSize())
        return self.getContents()[i][:]
    
    def addPoint(self,point):
        """Adds a COPY of point at the end of _contents.
        
        Parameter point: the point to add
        Precondition: point is a list of numbers (int or float), len(point) = _dimension.
        """
        assert is_point(point)
        assert len(point)==self._dimension
        self.getContents().append(point[:])

    def __str__(self):
        """Returns: String representation of the centroid of this cluster."""
        return str(self._contents)
    
    def __repr__(self):
        """Returns: Unambiguous representation of this cluster. """
        return str(self.__class__) + str(self)


class Cluster(object):
    """An instance is a cluster, a subset of the points in a dataset.
    
    A cluster is represented as a list of integers that give the indices
    in the dataset of the points contained in the cluster.  For instance,
    a cluster consisting of the points with indices 0, 4, and 5 in the
    dataset's data array would be represented by the index list [0,4,5].
    
    Instance Attributes:
        _dataset: the dataset this cluster is a subset of  [Dataset]
        
        _indices: the indices of this cluster's points in the dataset  [list of int]
        
        _centroid: the centroid of this cluster  [list of numbers]
    
    ADDITIONAL INVARIANTS:
        len(_centroid) == _dataset.getDimension()
        0 <= _indices[i] < _dataset.getSize(), for all 0 <= i < len(_indices)
    """
    
    def __init__(self, ds, centroid):
        """Initializer: Creates a new empty cluster with the given centroid.       
        
        Parameter ds: the Dataset for this cluster
        Precondition: ds is a instance of Dataset OR a subclass of Dataset
        
        Parameter centroid: the centroid point (which might not be a point in ds)
        Precondition: centroid is a list of numbers (int or float),
          len(centroid) = ds.getDimension()
        """
        assert isinstance(ds,Dataset)
        assert is_point(centroid)
        assert len(centroid)==ds.getDimension()

        self._indices = []
        self._centroid = centroid[:]
        self._dataset = ds

    def getCentroid(self):
        """Returns: the centroid of this cluster.
        """
        return self._centroid[:]
    
    def getIndices(self):
        """Returns: the indices of points in this cluster
        """
        return self._indices
    
    def addIndex(self, index):
        """Adds the given dataset index to this cluster.
        
        If the index is already in this cluster, then this method leaves the
        cluster unchanged.
        
        Parameter index: the index of the point to add
        Precondition: index is a valid index into this cluster's dataset.
        That is, index is an int in the range 0.._dataset.getSize()-1.
        """
        assert is_index(index,self._dataset.getSize())
        if index not in self.getIndices():
            self.getIndices().append(index)
    
    def clear(self):
        """Removes all points from this cluster, but leave the centroid unchanged.
        """
        self._indices = []
    
    def getContents(self):
        """Returns: a new list containing copies of the points in this cluster.
        """
        points = []
        for i in self.getIndices():
            points.append(self._dataset.getPoint(i))
        return points
    
    def distance(self, point):
        """Returns: The euclidean distance from point to this cluster's centroid.
        
        Parameter point: the point to compare to this cluster's centroid
        Precondition: point is a list of numbers (int or float),
          len(point) = _ds.getDimension()
        """
        assert is_point(point)
        assert len(point)==self._dataset.getDimension()
        total = 0
        for i in range(len(point)):
            total+=(point[i]-self.getCentroid()[i])**2
        return total**0.5
    
    def updateCentroid(self):
        """Returns: True if the centroid remains unchanged; False otherwise.
        
        This method recomputes the _centroid attribute of this cluster. The
        new _centroid attribute is the average of the points of _contents
        (To average a point, average each coordinate separately).  
        
        If there are no points in the cluster, the centroid. does not change.
        """
        old = self.getCentroid()[:]
        new =[]
        for i in range(len(old)):
            new.append(0.0)
        for point in self.getContents():
            for i in range(len(point)):
                new[i]+=point[i]
        for i in range(len(new)):
            new[i]=new[i]/len(self.getIndices())
        self._centroid = new
        change = numpy.allclose(self.getCentroid(),old)
        return change


    def __str__(self):
        """Returns: String representation of the centroid of this cluster."""
        return str(self._centroid)
    
    def __repr__(self):
        """Returns: Unambiguous representation of this cluster. """
        return str(self.__class__) + str(self)



class ClusterGroup(object):
    """An instance is a set of clusters of the points in a dataset.
    
    Instance Attributes:
        _dataset: the dataset which this is a clustering of     [Dataset]
        
        _clusters: the clusters in this clustering (not empty) list of Cluster]
    """
    
    def __init__(self, ds, k, seed_inds=None):
        """Initializer: Creates a clustering of the dataset ds into k clusters.
        
        The clusters are initialized by randomly selecting k different points
        from the database to be the centroids of the clusters.  If seed_inds
        is supplied, it is a list of indices into the dataset that specifies
        which points should be the initial cluster centroids.
        
        
        Parameter ds: the Dataset for this cluster group
        Precondition: ds is a instance of Dataset OR a subclass of Dataset
        
        Parameter k: The number of clusters (the k in k-means)
        Precondition: k is an int, 0 < k <= ds.getSize()
        
        Parameter seed_inds: the INDEXES of the points to start with
        Precondition: seed_inds is None, or a list of k valid indices into ds.
        """
        assert isinstance(ds,Dataset)
        assert type(k)==int and 0<k<=ds.getSize()
        assert seed_inds==None or is_valid_seed_inds(seed_inds,ds.getSize())

        self._dataset = ds
        self._clusters = []
        if seed_inds==None:
            for i in random.sample(ds.getContents(),k):
                self._clusters.append(Cluster(ds,i))
        else:
            for i in seed_inds:
                self._clusters.append(Cluster(ds,ds.getPoint(i)))


    def getClusters(self):
        """Returns: The list of clusters in this object.
        """ 
        return self._clusters

    def _nearest_cluster(self, point):
        """Returns: Cluster nearest to point
    
        This method uses the distance method of each Cluster to compute the distance 
        between point and the cluster centroid. It returns the Cluster that is 
        the closest.
        
        Ties are broken in favor of clusters occurring earlier in the list of 
        self._clusters.
        
        Parameter point: the point to match
        Precondition: point is a list of numbers (int or float),
          len(point) = self._dataset.getDimension().
        """
        assert is_point(point) and len(point)==self._dataset.getDimension()
        com_distances = []
        for i in self.getClusters():
            com_distances.append(i.distance(point))
        return self.getClusters()[com_distances.index(min(com_distances))]

    
    def _partition(self):
        """Repartitions the dataset so each point is in exactly one Cluster.
        """
        for i in self.getClusters():
            i._indices = []
        for point in self._dataset.getContents():
            new_cluster = self._nearest_cluster(point)
            new_cluster.addIndex(self._dataset.getContents().index(point))
    
    def _update(self):
        """Returns:True if all centroids are unchanged after an update; False otherwise.
        
        This method first updates the centroids of all clusters'.  When it is done, it
        checks whether any of them have changed. It then returns the appropriate value.
        """
        old_centroids = []
        for i in self.getClusters():
            old_centroids.append(i.getCentroid())
            i.updateCentroid()
        for x in range(len(old_centroids)):
            if old_centroids[x]!=self.getClusters()[x].getCentroid():
                return False
        return True
            
    
    def step(self):
        """Returns: True if the algorithm converges after one step; False otherwise.
        
        This method performs one cycle of the k-means algorithm. It then checks if
        the algorithm has converged and returns True or False
        """
        self._partition()
        return self._update()
    
    def run(self, maxstep):
        """Continues clustering until either it converges or reaches maxstep steps.
        
        The stopping condition (convergence, maxsteps) is whichever comes first.
        
        Precondition maxstep: Maximum number of steps before giving up
        Precondition: maxstep is int >= 0.
        """
        assert type(maxstep)==int and maxstep>=0

        i=0
        while i<maxstep:
            self.step()
            i+=1
    
    def __str__(self):
        """Returns: String representation of the centroid of this cluster."""
        return str(self._clusters)
    
    def __repr__(self):
        """Returns: Unambiguous representation of this cluster. """
        return str(self.__class__) + str(self)

