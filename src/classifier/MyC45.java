/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier;

import java.util.Enumeration;
import weka.classifiers.AbstractClassifier;
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
        return 0.0;
    }
    
    private double computeGainRatio(Instances data, Attribute attr) {
    // Info gain/Split information
        return 0.0;
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
    
    private void replaceMissingValues(Instances data) throws Exception {
        Instances replacedData;
        ReplaceMissingValues filter = new ReplaceMissingValues();

        filter.setInputFormat(data);
        replacedData = Filter.useFilter(data, filter);
    }
    
    private void splitContinuousValue(Instances data) {
    // didescretize aja gabole ya
    // defaultnya pake binary split yang di binc45 something itu
    }
    
    private void pruneTree(Instances data) {
        
    }
    
    // Error handling masih belom gt ngerti soalnya ada beberapa jenisnya
}
