/**
 * Copyright 2009 Rednaxela
 * 
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 * 
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 * 
 *    1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 
 *    2. This notice may not be removed or altered from any source
 *    distribution.
 */
/***
 * 
 * SH: Added a "mirror tree" mechanism to enable fast and safe multi-thread query
 * 
 */

//package ags.utils;
 
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
 
/**
 * An efficient well-optimized kd-tree
 * 
 * @author Rednaxela
 */
public abstract class KdTree2<T> {
    // Static variables
    private static final int           bucketSize = 24;
 
    // All types
    private final int                  dimensions;
    private final KdTree2<T>            parent;
 
    // Root only
    private final LinkedList<double[]> locationStack;
    private final Integer              sizeLimit;
 
    // Leaf only
    private double[][]                 locations;
    private Object[]                   data;
    private int                        locationCount;
 
    // Stem only
    private KdTree2<T>                  left, right;
    private int                        splitDimension;
    private double                     splitValue;
 
    // Bounds
    private double[]                   minLimit, maxLimit;
    private boolean                    singularity;
 
 
    /**
     * Construct a KdTree with a given number of dimensions and a limit on
     * maxiumum size (after which it throws away old points)
     */
    private KdTree2(int dimensions, Integer sizeLimit) {
        this.dimensions = dimensions;
 
        // Init as leaf
        this.locations = new double[bucketSize][];
        this.data = new Object[bucketSize];
        this.locationCount = 0;
        this.singularity = true;
 
        // Init as root
        this.parent = null;
        this.sizeLimit = sizeLimit;
        if (sizeLimit != null) {
            this.locationStack = new LinkedList<double[]>();
        }
        else {
            this.locationStack = null;
        }
    }
 
    /**
     * Constructor for child nodes. Internal use only.
     */
    private KdTree2(KdTree2<T> parent, boolean right) {
        this.dimensions = parent.dimensions;
 
        // Init as leaf
        this.locations = new double[Math.max(bucketSize, parent.locationCount)][];
        this.data = new Object[Math.max(bucketSize, parent.locationCount)];
        this.locationCount = 0;
        this.singularity = true;
 
        // Init as non-root
        this.parent = parent;
        this.locationStack = null;
        this.sizeLimit = null;
    }
 
    /**
     * Get the number of points in the tree
     */
    public int size() {
        return locationCount;
    }
 
    /**
     * Add a point and associated value to the tree
     */
    public void addPoint(double[] location, T value) {
        KdTree2<T> cursor = this;
 
        while (cursor.locations == null || cursor.locationCount >= cursor.locations.length) {
            if (cursor.locations != null) {
                cursor.splitDimension = cursor.findWidestAxis();
                cursor.splitValue = (cursor.minLimit[cursor.splitDimension] + cursor.maxLimit[cursor.splitDimension]) * 0.5;
 
                // Never split on infinity or NaN
                if (cursor.splitValue == Double.POSITIVE_INFINITY) {
                    cursor.splitValue = Double.MAX_VALUE;
                }
                else if (cursor.splitValue == Double.NEGATIVE_INFINITY) {
                    cursor.splitValue = -Double.MAX_VALUE;
                }
                else if (Double.isNaN(cursor.splitValue)) {
                    cursor.splitValue = 0;
                }
 
                // Don't split node if it has no width in any axis. Double the
                // bucket size instead
                if (cursor.minLimit[cursor.splitDimension] == cursor.maxLimit[cursor.splitDimension]) {
                    double[][] newLocations = new double[cursor.locations.length * 2][];
                    System.arraycopy(cursor.locations, 0, newLocations, 0, cursor.locationCount);
                    cursor.locations = newLocations;
                    Object[] newData = new Object[newLocations.length];
                    System.arraycopy(cursor.data, 0, newData, 0, cursor.locationCount);
                    cursor.data = newData;
                    break;
                }
 
                // Don't let the split value be the same as the upper value as
                // can happen due to rounding errors!
                if (cursor.splitValue == cursor.maxLimit[cursor.splitDimension]) {
                    cursor.splitValue = cursor.minLimit[cursor.splitDimension];
                }
 
                // Create child leaves
                KdTree2<T> left = new ChildNode(cursor, false);
                KdTree2<T> right = new ChildNode(cursor, true);
 
                // Move locations into children
                for (int i = 0; i < cursor.locationCount; i++) {
                    double[] oldLocation = cursor.locations[i];
                    Object oldData = cursor.data[i];
                    if (oldLocation[cursor.splitDimension] > cursor.splitValue) {
                        // Right
                        right.locations[right.locationCount] = oldLocation;
                        right.data[right.locationCount] = oldData;
                        right.locationCount++;
                        right.extendBounds(oldLocation);
                    }
                    else {
                        // Left
                        left.locations[left.locationCount] = oldLocation;
                        left.data[left.locationCount] = oldData;
                        left.locationCount++;
                        left.extendBounds(oldLocation);
                    }
                }
 
                // Make into stem
                cursor.left = left;
                cursor.right = right;
                cursor.locations = null;
                cursor.data = null;
            }
 
            cursor.locationCount++;
            cursor.extendBounds(location);
 
            if (location[cursor.splitDimension] > cursor.splitValue) {
                cursor = cursor.right;
            }
            else {
                cursor = cursor.left;
            }
        }
 
        cursor.locations[cursor.locationCount] = location;
        cursor.data[cursor.locationCount] = value;
        cursor.locationCount++;
        cursor.extendBounds(location);
 
        if (this.sizeLimit != null) {
            this.locationStack.add(location);
            if (this.locationCount > this.sizeLimit) {
                this.removeOld();
            }
        }
    }
 
    /**
     * Extends the bounds of this node do include a new location
     */
    private final void extendBounds(double[] location) {
        if (minLimit == null) {
            minLimit = new double[dimensions];
            System.arraycopy(location, 0, minLimit, 0, dimensions);
            maxLimit = new double[dimensions];
            System.arraycopy(location, 0, maxLimit, 0, dimensions);
            return;
        }
 
        for (int i = 0; i < dimensions; i++) {
            if (Double.isNaN(location[i])) {
                minLimit[i] = Double.NaN;
                maxLimit[i] = Double.NaN;
                singularity = false;
            }
            else if (minLimit[i] > location[i]) {
                minLimit[i] = location[i];
                singularity = false;
            }
            else if (maxLimit[i] < location[i]) {
                maxLimit[i] = location[i];
                singularity = false;
            }
        }
    }
 
    /**
     * Find the widest axis of the bounds of this node
     */
    private final int findWidestAxis() {
        int widest = 0;
        double width = (maxLimit[0] - minLimit[0]) * getAxisWeightHint(0);
        if (Double.isNaN(width)) width = 0;
        for (int i = 1; i < dimensions; i++) {
            double nwidth = (maxLimit[i] - minLimit[i]) * getAxisWeightHint(i);
            if (Double.isNaN(nwidth)) nwidth = 0;
            if (nwidth > width) {
                widest = i;
                width = nwidth;
            }
        }
        return widest;
    }
 
    /**
     * Remove the oldest value from the tree. Note: This cannot trim the bounds
     * of nodes, nor empty nodes, and thus you can't expect it to perfectly
     * preserve the speed of the tree as you keep adding.
     */
    private void removeOld() {
        double[] location = this.locationStack.removeFirst();
        KdTree2<T> cursor = this;
 
        // Find the node where the point is
        while (cursor.locations == null) {
            if (location[cursor.splitDimension] > cursor.splitValue) {
                cursor = cursor.right;
            }
            else {
                cursor = cursor.left;
            }
        }
 
        for (int i = 0; i < cursor.locationCount; i++) {
            if (cursor.locations[i] == location) {
                System.arraycopy(cursor.locations, i + 1, cursor.locations, i, cursor.locationCount - i - 1);
                cursor.locations[cursor.locationCount-1] = null;
                System.arraycopy(cursor.data, i + 1, cursor.data, i, cursor.locationCount - i - 1);
                cursor.data[cursor.locationCount-1] = null;
                do {
                    cursor.locationCount--;
                    cursor = cursor.parent;
                } while (cursor != null);
                return;
            }
        }
        // If we got here... we couldn't find the value to remove. Weird...
    }
 
    /**
     * Enumeration representing the status of a node during the running
     */
    private static enum Status {
        NONE, LEFTVISITED, RIGHTVISITED, ALLVISITED
    }
 
    /**
     * Stores a distance and value to output
     */
    public static class Entry<T> {
        public final double distance;
        public final T      value;
 
        private Entry(double distance, T value) {
            this.distance = distance;
            this.value = value;
        }
    }
 
    /**
     * Calculates the nearest 'count' points to 'location'
     */
    @SuppressWarnings("unchecked")
    public List<Entry<T>> nearestNeighbor(double[] location, int count, boolean sequentialSorting) {
    	MirrorTree cursor=new MirrorTree(this,null);
        cursor.status = Status.NONE;
        double range = Double.POSITIVE_INFINITY;
        ResultHeap resultHeap = new ResultHeap(count);
 
        do {
            if (cursor.status == Status.ALLVISITED) {
                // At a fully visited part. Move up the tree
                cursor = cursor.parent;
                continue;
            }
 
            if (cursor.status == Status.NONE && cursor.node.locations != null) {
                // At a leaf. Use the data.
                if (cursor.node.locationCount > 0) {
                    if (cursor.node.singularity) {
                        double dist = pointDist(cursor.node.locations[0], location);
                        if (dist <= range) {
                            for (int i = 0; i < cursor.node.locationCount; i++) {
                                resultHeap.addValue(dist, cursor.node.data[i]);
                            }
                        }
                    }
                    else {
                        for (int i = 0; i < cursor.node.locationCount; i++) {
                            double dist = pointDist(cursor.node.locations[i], location);
                            resultHeap.addValue(dist, cursor.node.data[i]);
                        }
                    }
                    range = resultHeap.getMaxDist();
                }
 
                if (cursor.parent == null) {
                    break;
                }
                cursor = cursor.parent;
                continue;
            }
 
            // Going to descend
            MirrorTree nextCursor = null;
            if (cursor.status == Status.NONE) {
                // At a fresh node, descend the most probably useful direction
                if (location[cursor.node.splitDimension] > cursor.node.splitValue) {
                    // Descend right
                    nextCursor = new MirrorTree(cursor.node.right,cursor);
                    cursor.status = Status.RIGHTVISITED;
                }
                else {
                    // Descend left;
                    nextCursor = new MirrorTree(cursor.node.left,cursor);
                    cursor.status = Status.LEFTVISITED;
                }
            }
            else if (cursor.status == Status.LEFTVISITED) {
                // Left node visited, descend right.
                nextCursor = new MirrorTree(cursor.node.right,cursor);
                cursor.status = Status.ALLVISITED;
            }
            else if (cursor.status == Status.RIGHTVISITED) {
                // Right node visited, descend left.
                nextCursor = new MirrorTree(cursor.node.left,cursor);
                cursor.status = Status.ALLVISITED;
            }
 
            // Check if it's worth descending. Assume it is if it's sibling has
            // not been visited yet.
            if (cursor.status == Status.ALLVISITED) {
                if (nextCursor.node.locationCount == 0
                        || (!nextCursor.node.singularity && pointRegionDist(location, nextCursor.node.minLimit,
                                nextCursor.node.maxLimit) > range)) {
                    continue;
                }
            }
 
            // Descend down the tree
            cursor = nextCursor;
            cursor.status = Status.NONE;
        } while (cursor.parent != null || cursor.status != Status.ALLVISITED);
 
        ArrayList<Entry<T>> results = new ArrayList<Entry<T>>(resultHeap.values);
        if (sequentialSorting) {
            while (resultHeap.values > 0) {
                resultHeap.removeLargest();
                results.add(new Entry<T>(resultHeap.removedDist, (T)resultHeap.removedData));
            }
        }
        else {
            for (int i = 0; i < resultHeap.values; i++) {
                results.add(new Entry<T>(resultHeap.distance[i], (T)resultHeap.data[i]));
            }
        }
 
        return results;
    }
 
    // Override in subclasses
    protected abstract double pointDist(double[] p1, double[] p2);
 
    protected abstract double pointRegionDist(double[] point, double[] min, double[] max);
 
    protected double getAxisWeightHint(int i) {
        return 1.0;
    }
 
    /**
     * Internal class for child nodes
     */
    private class ChildNode extends KdTree2<T> {
        private ChildNode(KdTree2<T> parent, boolean right) {
            super(parent, right);
        }
 
        // Distance measurements are always called from the root node
        protected double pointDist(double[] p1, double[] p2) {
            throw new IllegalStateException();
        }
 
        protected double pointRegionDist(double[] point, double[] min, double[] max) {
            throw new IllegalStateException();
        }
    }
 
    /**
     * Class for tree with Weighted Squared Euclidean distancing
     */
    public static class WeightedSqrEuclid<T> extends KdTree2<T> {
        private double[] weights;
 
        public WeightedSqrEuclid(int dimensions, Integer sizeLimit) {
            super(dimensions, sizeLimit);
            this.weights = new double[dimensions];
            Arrays.fill(this.weights, 1.0);
        }
 
        public void setWeights(double[] weights) {
            this.weights = weights;
        }
 
        protected double getAxisWeightHint(int i) {
            return weights[i];
        }
 
        protected double pointDist(double[] p1, double[] p2) {
            double d = 0;
 
            for (int i = 0; i < p1.length; i++) {
                double diff = (p1[i] - p2[i]) * weights[i];
                if (!Double.isNaN(diff)) {
                    d += diff * diff;
                }
            }
 
            return d;
        }
 
        protected double pointRegionDist(double[] point, double[] min, double[] max) {
            double d = 0;
 
            for (int i = 0; i < point.length; i++) {
                double diff = 0;
                if (point[i] > max[i]) {
                    diff = (point[i] - max[i]) * weights[i];
                }
                else if (point[i] < min[i]) {
                    diff = (point[i] - min[i]) * weights[i];
                }
 
                if (!Double.isNaN(diff)) {
                    d += diff * diff;
                }
            }
 
            return d;
        }
    }
 
    /**
     * Class for tree with Unweighted Squared Euclidean distancing
     */
    public static class SqrEuclid<T> extends KdTree2<T> {
        public SqrEuclid(int dimensions, Integer sizeLimit) {
            super(dimensions, sizeLimit);
        }
 
        protected double pointDist(double[] p1, double[] p2) {
            double d = 0;
 
            for (int i = 0; i < p1.length; i++) {
                double diff = (p1[i] - p2[i]);
                if (!Double.isNaN(diff)) {
                    d += diff * diff;
                }
            }
 
            return d;
        }
 
        protected double pointRegionDist(double[] point, double[] min, double[] max) {
            double d = 0;
 
            for (int i = 0; i < point.length; i++) {
                double diff = 0;
                if (point[i] > max[i]) {
                    diff = (point[i] - max[i]);
                }
                else if (point[i] < min[i]) {
                    diff = (point[i] - min[i]);
                }
 
                if (!Double.isNaN(diff)) {
                    d += diff * diff;
                }
            }
 
            return d;
        }
    }
 
    /**
     * Class for tree with Weighted Manhattan distancing
     */
    public static class WeightedManhattan<T> extends KdTree2<T> {
        private double[] weights;
 
        public WeightedManhattan(int dimensions, Integer sizeLimit) {
            super(dimensions, sizeLimit);
            this.weights = new double[dimensions];
            Arrays.fill(this.weights, 1.0);
        }
 
        public void setWeights(double[] weights) {
            this.weights = weights;
        }
 
        protected double getAxisWeightHint(int i) {
            return weights[i];
        }
 
        protected double pointDist(double[] p1, double[] p2) {
            double d = 0;
 
            for (int i = 0; i < p1.length; i++) {
                double diff = (p1[i] - p2[i]);
                if (!Double.isNaN(diff)) {
                    d += ((diff < 0) ? -diff : diff) * weights[i];
                }
            }
 
            return d;
        }
 
        protected double pointRegionDist(double[] point, double[] min, double[] max) {
            double d = 0;
 
            for (int i = 0; i < point.length; i++) {
                double diff = 0;
                if (point[i] > max[i]) {
                    diff = (point[i] - max[i]);
                }
                else if (point[i] < min[i]) {
                    diff = (min[i] - point[i]);
                }
 
                if (!Double.isNaN(diff)) {
                    d += diff * weights[i];
                }
            }
 
            return d;
        }
    }
 
    /**
     * Class for tree with Manhattan distancing
     */
    public static class Manhattan<T> extends KdTree2<T> {
        public Manhattan(int dimensions, Integer sizeLimit) {
            super(dimensions, sizeLimit);
        }
 
        protected double pointDist(double[] p1, double[] p2) {
            double d = 0;
 
            for (int i = 0; i < p1.length; i++) {
                double diff = (p1[i] - p2[i]);
                if (!Double.isNaN(diff)) {
                    d += (diff < 0) ? -diff : diff;
                }
            }
 
            return d;
        }
 
        protected double pointRegionDist(double[] point, double[] min, double[] max) {
            double d = 0;
 
            for (int i = 0; i < point.length; i++) {
                double diff = 0;
                if (point[i] > max[i]) {
                    diff = (point[i] - max[i]);
                }
                else if (point[i] < min[i]) {
                    diff = (min[i] - point[i]);
                }
 
                if (!Double.isNaN(diff)) {
                    d += diff;
                }
            }
 
            return d;
        }
    }
 
    /**
     * Class for tracking up to 'size' closest values
     */
    private static class ResultHeap {
        private final Object[] data;
        private final double[] distance;
        private final int      size;
        private int            values;
        public Object          removedData;
        public double          removedDist;
 
        public ResultHeap(int size) {
            this.data = new Object[size];
            this.distance = new double[size];
            this.size = size;
            this.values = 0;
        }
 
        public void addValue(double dist, Object value) {
            // If there is still room in the heap
            if (values < size) {
                // Insert new value at the end
                data[values] = value;
                distance[values] = dist;
                upHeapify(values);
                values++;
            }
            // If there is no room left in the heap, and the new entry is lower
            // than the max entry
            else if (dist < distance[0]) {
                // Replace the max entry with the new entry
                data[0] = value;
                distance[0] = dist;
                downHeapify(0);
            }
        }
 
        public void removeLargest() {
            if (values == 0) {
                throw new IllegalStateException();
            }
 
            removedData = data[0];
            removedDist = distance[0];
            values--;
            data[0] = data[values];
            distance[0] = distance[values];
            downHeapify(0);
        }
 
        private void upHeapify(int c) {
            for (int p = (c - 1) / 2; c != 0 && distance[c] > distance[p]; c = p, p = (c - 1) / 2) {
                Object pData = data[p];
                double pDist = distance[p];
                data[p] = data[c];
                distance[p] = distance[c];
                data[c] = pData;
                distance[c] = pDist;
            }
        }
 
        private void downHeapify(int p) {
            for (int c = p * 2 + 1; c < values; p = c, c = p * 2 + 1) {
                if (c + 1 < values && distance[c] < distance[c + 1]) {
                    c++;
                }
                if (distance[p] < distance[c]) {
                    // Swap the points
                    Object pData = data[p];
                    double pDist = distance[p];
                    data[p] = data[c];
                    distance[p] = distance[c];
                    data[c] = pData;
                    distance[c] = pDist;
                }
                else {
                    break;
                }
            }
        }
 
        public double getMaxDist() {
            if (values < size) {
                return Double.POSITIVE_INFINITY;
            }
            return distance[0];
        }
    }
    
    class MirrorTree {
    	KdTree2<T> node;
    	MirrorTree parent;
    	Status status;
    	public MirrorTree(KdTree2<T> n,MirrorTree p) {
    		node=n;
    		parent=p;
    	}
    }
}
