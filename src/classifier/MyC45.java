/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier;

import java.util.Enumeration;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.GainRatioSplitCrit;
import weka.classifiers.trees.j48.InfoGainSplitCrit;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 *
 * @author BerviantoLeoPratama
 * SekarAnglilaHapsari
 */
public class MyC45 extends AbstractClassifier {

    private MyC45[] node;

    /**
     * Class value if node is leaf.
     */
    private double m_ClassValue;

    private double[] m_infoGainData;

    /**
     * Class distribution if node is leaf.
     */
    private double[] m_Distribution;

    /**
     * Attribute used for splitting.
     */
    private Attribute m_Attribute;

    /**
     * Class attribute of dataset.
     */
    private Attribute m_ClassAttribute;
    
    private double m_splitPoint = Double.MAX_VALUE;
    
    private double m_sumOfWeights;
    
    private double m_infoGain;
    
    private static InfoGainSplitCrit infoGainCrit = new InfoGainSplitCrit();
    
    private static GainRatioSplitCrit gainRatioCrit = new GainRatioSplitCrit();
    
    private double m_gainRatio = 0;
    
    private int m_minNoObj = 2;

    /**
     * Computes the entropy of a dataset.
     *
     * @param data the data for which entropy is to be computed
     * @return the entropy of the data's class distribution
     * @throws Exception if computation fails
     */
    private double computeEntropy(Instances data) throws Exception {

        double[] classCounts = new double[data.numClasses()];
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classCounts[(int) inst.classValue()]++;
        }
        double entropy = 0;
        for (int j = 0; j < data.numClasses(); j++) {
            if (classCounts[j] > 0) {
                entropy -= classCounts[j] * Utils.log2(classCounts[j]);
            }
        }
        entropy /= (double) data.numInstances();
        return entropy + Utils.log2(data.numInstances());
    }

    /**
     * Splits a dataset according to the values of a nominal attribute.
     *
     * @param data the data which is to be split
     * @param att the attribute to be used for splitting
     * @return the sets of instances produced by the split
     */
    private Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];
        for (int j = 0; j < att.numValues(); j++) {
            splitData[j] = new Instances(data, data.numInstances());
        }
        Enumeration<Instance> instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = instEnum.nextElement();
            splitData[(int) inst.value(att)].add(inst);
        }
        for (Instances splitData1 : splitData) {
            splitData1.compactify();
        }
        return splitData;
    }

    private double computeInfoGain(Instances data, Attribute attr) throws Exception {
        double infoGain = computeEntropy(data);
        Instances[] splitedInstances = splitData(data, attr);

        for (int i = 0; i < attr.numValues(); i++) {
            if (splitedInstances[i].numInstances() > 0) {
                infoGain -= ((splitedInstances[i].numInstances() / (double) data.numInstances()) * computeEntropy(splitedInstances[i]));
            }
        }

        return infoGain;
    }
    
    private double computeSplitInformation(Instances data, Attribute attr) {
    // Basically ngitung entropi kelas
        Instances[] splitedInstances = splitData(data, attr);
        double splitInfo = 0.0;
                
        for (int i = 0; i < attr.numValues(); i++) {
            if (splitedInstances[i].numInstances() > 0) {
                double attrRatio = splitedInstances[i].numInstances() / (double) data.numInstances() ;
                splitInfo -=  attrRatio * Utils.log2(attrRatio);
            }
        }
        
        return splitInfo;
    }
    
    private double computeGainRatio(Instances data, Attribute attr) throws Exception {
    // Info gain/Split information
        double infoGain = computeInfoGain(data, attr);
        double splitInfo = computeSplitInformation(data, attr);
        if (splitInfo == 0.0) {
            return Double.NEGATIVE_INFINITY;
        } else {
            return infoGain / splitInfo;
        }
    }
    
    private double[] enumerateInfoGain(Instances data)
            throws Exception {
        double[] infoGains = new double[data.numAttributes()];
        Enumeration<Attribute> attEnum = data.enumerateAttributes();
        while (attEnum.hasMoreElements()) {
            Attribute att = attEnum.nextElement();
            infoGains[att.index()] = computeInfoGain(data, att);
        }
        return infoGains;
    }

    private double[] enumerateGainRatio(Instances data)
            throws Exception {
        double[] gainRatios = new double[data.numAttributes()];
        Enumeration<Attribute> attEnum = data.enumerateAttributes();
        while (attEnum.hasMoreElements()) {
            Attribute att = attEnum.nextElement();
            gainRatios[att.index()] = computeGainRatio(data, att);
        }
        return gainRatios;
    }
    private double mostCommonValue(Instances example) {
        int[] dataClass = new int[example.numClasses()];
        Enumeration<Instance> inst = example.enumerateInstances();
        while (inst.hasMoreElements()) {
            Instance instance = inst.nextElement();
            dataClass[(int) instance.classValue()] += 1;
        }
        int mostCommonValue = 0;
        int max = 0;
        for (int i = 0; i < example.numClasses(); i++) {
            if (dataClass[i] > max) {
                max = dataClass[i];
                mostCommonValue = i;
            }
        }
        return (double) mostCommonValue;
    }

    private void makeTree(Instances instances) throws Exception {

        // Check if no instances have reached this node.
        if (instances.numInstances() == 0) {
            m_Attribute = null;
            m_ClassValue = Utils.missingValue();
            m_Distribution = new double[instances.numClasses()];
            return;
        }
        
        this.m_infoGainData = enumerateInfoGain(instances);
        this.m_Attribute = instances.attribute(Utils.maxIndex(m_infoGainData));
        if (Utils.eq(this.m_infoGainData[m_Attribute.index()], 0)) {
            // Max Info Gain = 0
            m_Attribute = null;
            m_Distribution = new double[instances.numClasses()];
            Enumeration instEnum = instances.enumerateInstances();
            while (instEnum.hasMoreElements()) {
                Instance inst = (Instance) instEnum.nextElement();
                m_Distribution[(int) inst.classValue()]++;
            }
            Utils.normalize(m_Distribution);
            m_ClassValue = mostCommonValue(instances);
            m_ClassAttribute = instances.classAttribute();
        } else {
            Instances[] exampleVi = splitData(instances, m_Attribute);
            this.node = new MyC45[m_Attribute.numValues()];
            for (int j = 0; j < m_Attribute.numValues(); j++) {
                this.node[j] = new MyC45();
                this.node[j].makeTree(exampleVi[j]);
            }
        }
    }

    //Masih harus di edit
    @Override
    public void buildClassifier(Instances i) throws Exception {
        i = new Instances(i);
        i.deleteWithMissingClass();
        i = replaceMissingValues(i);
        makeTree(i);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if (this.m_ClassAttribute == null) {
            return this.m_ClassValue;
        } else {
            return this.node[(int) instance.value(this.m_ClassAttribute)].classifyInstance(instance);
        }
    }

    /**
     * Computes class distribution for instance using decision tree.
     *
     * @param instance the instance for which distribution is to be computed
     * @return the class distribution for the given instance
     */
    @Override
    public double[] distributionForInstance(Instance instance) {
        if (m_Attribute == null) {
            return m_Distribution;
        } else {
            return this.node[(int) instance.value(m_Attribute)].
                    distributionForInstance(instance);
        }
    }

    @Override
    public String toString() {

        if ((m_Distribution == null) && (this.node == null)) {
            return "C45: No model built yet.";
        }
        return "C45\n\n" + toString(0);
    }

    /**
     * Outputs a tree at a certain level.
     *
     * @param level the level at which the tree is to be printed
     * @return the tree as string at the given level
     */
    private String toString(int level) {

        StringBuilder text = new StringBuilder();

        //Masih harus di benerin
        if (m_Attribute == null) {
            if (Utils.isMissingValue(m_ClassValue)) {
                text.append(": null");
            } else {
                text.append(": ").append(m_ClassAttribute.value((int) m_ClassValue));
            }
        } else {
            for (int j = 0; j < m_Attribute.numValues(); j++) {
                text.append("\n");
                for (int i = 0; i < level; i++) {
                    text.append("|  ");
                }
                text.append(m_Attribute.name()).append(" = ").append(m_Attribute.value(j));
                text.append(this.node[j].toString(level + 1));
            }
        }
        return text.toString();
    }
    
    private Instances replaceMissingValues(Instances data) throws Exception {
        Instances replacedData;
        ReplaceMissingValues filter = new ReplaceMissingValues();

        filter.setInputFormat(data);
        replacedData = Filter.useFilter(data, filter);
        
        return replacedData;
    }
    
    private void splitContinuousValue(Instances data) throws Exception{
        //Kamus
        int firstMiss;
        int next = 1;
        int last = 0;
        int splitIndex = -1;
        double currentInfoGain;
        double defaultEnt;
        double minSplit;
        Instance instance;
        int i;
        Distribution m_dist;
        m_sumOfWeights = data.sumOfWeights();
        int m_index = 0;
        m_infoGain = 0;

        //Algoritma
        // Current attribute is a numeric attribute.
        m_dist = new Distribution(2,data.numClasses());

        // Only Instances with known values are relevant.
        Enumeration enu = data.enumerateInstances();
        i = 0;
        while (enu.hasMoreElements()) {
          instance = (Instance) enu.nextElement();
          if (instance.isMissing(m_Attribute.index())) {
            break;
          }
          m_dist.add(1, instance);
          i++;
        }
        firstMiss = i;

        // Compute minimum number of Instances required in each subset.
        minSplit =  0.1 * (m_dist.total()) / ((double)data.numClasses());
        if (Utils.smOrEq(minSplit, m_minNoObj)) {
          minSplit = m_minNoObj;
        } else {
          if (Utils.gr(minSplit, 25)) {
            minSplit = 25;
          }
        }

        // Enough Instances with known values?
        if (Utils.sm((double)firstMiss, (2 * minSplit))) {
          return;
        }

        // Compute values of criteria for all possible split
        // indices.
        defaultEnt = infoGainCrit.oldEnt(m_dist);
        while (next < firstMiss) {
          if (data.instance(next-1).value(m_Attribute.index()) + 1e-5 < data.instance(next).value(m_Attribute.index())) { 

            // Move class values for all Instances up to next possible split point.
            m_dist.shiftRange(1,0,data,last,next);

            // Check if enough Instances in each subset and compute
            // values for criteria.
            if (Utils.grOrEq(m_dist.perBag(0),minSplit) && Utils.grOrEq(m_dist.perBag(1),minSplit)) {
              currentInfoGain = infoGainCrit.splitCritValue(m_dist,m_sumOfWeights,defaultEnt);
              if (Utils.gr(currentInfoGain, m_infoGain)) {
                m_infoGain = currentInfoGain;
                splitIndex = next-1;
              }
              m_index++;
            }
            last = next;
          }
          next++;
        }

        // Was there any useful split?
        if (m_index == 0) {
          return;
        }

        // Compute modified information gain for best split.
        m_infoGain = m_infoGain - (Utils.log2(m_index)/m_sumOfWeights);
        if (Utils.smOrEq(m_infoGain,0)) {
          return;
        }

        // Set instance variables' values to values for best split.
        int m_numSubsets = 2;
        m_splitPoint =  (data.instance(splitIndex+1).value(m_Attribute.index()) + data.instance(splitIndex).value(m_Attribute.index()))/2;

        // In case we have a numerical precision problem we need to choose the
        // smaller value
        if (m_splitPoint == data.instance(splitIndex + 1).value(m_Attribute.index())) {
          m_splitPoint = data.instance(splitIndex).value(m_Attribute.index());
        }

        // Restore distributioN for best split.
        m_dist = new Distribution(2,data.numClasses());
        m_dist.addRange(0,data,0,splitIndex+1);
        m_dist.addRange(1,data,splitIndex+1,firstMiss);

        // Compute modified gain ratio for best split.
        m_gainRatio = gainRatioCrit.splitCritValue(m_dist, m_sumOfWeights, m_infoGain);
    }
    
    private void pruneTree(Instances data) {
        
    }
    
    // Error handling masih belom gt ngerti soalnya ada beberapa jenisnya
}
