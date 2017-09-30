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

/**
 *
 * @author BerviantoLeoPratama
 * SekarAnglilaHapsari
 */
public class MyC45 extends AbstractClassifier {

    @Override
    public void buildClassifier(Instances data) throws Exception {
        data = new Instances(data);
        data.deleteWithMissingClass();
        makeTree(data);
    }
    
    private void makeTree(Instances data) throws Exception {
        
    }
    
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
    
    private double[] getMaxInfoGain(Instances data) throws Exception {
        double[] maxIG = new double[2];
        double infoGain;
        double maxInfoGain = 0;
        double i = 0;
        
        //Algoritma
        Enumeration<Attribute> en = data.enumerateAttributes();
        while (en.hasMoreElements()) {
                Attribute attr = (Attribute) en.nextElement();
                infoGain = computeInfoGain(data, attr);
                if (maxInfoGain < infoGain) {
                        maxInfoGain = infoGain;
                        i = attr.index();
                }
        }
        
        maxIG[0] = i;
        maxIG[1] = maxInfoGain;
        return maxIG;
    }
    
}
