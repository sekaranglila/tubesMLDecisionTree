/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.id3;

import java.util.Enumeration;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author BerviantoLeoPratama
 */
public class MyID3 extends AbstractClassifier {

    private Node tree;
    private int nodeCounter;
    private Instances data;
    private Instances examplevi;

    @Override
    public void buildClassifier(Instances i) throws Exception {
        nodeCounter = 0;
        data = new Instances(i);
        examplevi = new Instances(i);
        tree = makeTree();
    }

    private boolean isAllExamplesPositive() {
        boolean positive = false;
        Enumeration<Instance> insEnum = data.enumerateInstances();
        if (insEnum.hasMoreElements()) {
            double min, max;
            Instance inisial = insEnum.nextElement();
            min = inisial.classValue();
            max = inisial.classValue();
            while (insEnum.hasMoreElements()) {
                Instance nextInst = insEnum.nextElement();
                double classValue = nextInst.classValue();
                if (classValue > max) {
                    max = classValue;
                }
                if (classValue < min) {
                    min = classValue;
                }
            }
            if (min == max) {
                positive = true;
            }
        }
        return positive;
    }

    private double computeInfoGain(Instances data, Attribute att) 
    throws Exception {

    double infoGain = computeEntropy(data);
    Instances[] splitData = splitData(data, att);
    for (int j = 0; j < att.numValues(); j++) {
      if (splitData[j].numInstances() > 0) {
        infoGain -= ((double) splitData[j].numInstances() /
                     (double) data.numInstances()) *
          computeEntropy(splitData[j]);
      }
    }
    return infoGain;
  }

  /**
   * Computes the entropy of a dataset.
   * 
   * @param data the data for which entropy is to be computed
   * @return the entropy of the data's class distribution
   * @throws Exception if computation fails
   */
  private double computeEntropy(Instances data) throws Exception {

    double [] classCounts = new double[data.numClasses()];
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
    Enumeration instEnum = data.enumerateInstances();
    while (instEnum.hasMoreElements()) {
      Instance inst = (Instance) instEnum.nextElement();
      splitData[(int) inst.value(att)].add(inst);
    }
      for (Instances splitData1 : splitData) {
          splitData1.compactify();
      }
    return splitData;
  }
    
    private Attribute findBestAttribute() throws Exception {
        double bestGain = -10;
        Enumeration<Attribute> attr = data.enumerateAttributes();
        if (attr.hasMoreElements()) {
            Attribute best = attr.nextElement();
            while (attr.hasMoreElements()) {
                Attribute attribute = attr.nextElement();
                double gain = computeInfoGain(data, attribute);
                if (gain >= bestGain) {
                    bestGain = gain;
                    best = attribute;
                }
            }
            return best;
        }
        return null;
    }

    private void selectSubset(Attribute a, Object o) {
        Instances subset = new Instances(this.examplevi, this.examplevi.numAttributes()); 
        Enumeration<Instance> ins = this.examplevi.enumerateInstances();
         while(ins.hasMoreElements()) {
             Instance instance = ins.nextElement();
             if (instance.attribute(a.index()).indexOfValue(String.valueOf(o)) >= 0) {
                 subset.add(instance);
             }
         }
         this.examplevi = new Instances(subset);
    }
    
    private double mostCommonValue(Instances example) {
        return 0.0;
    }
    
    private Node makeTree() throws Exception {
        Instances example = new Instances(this.examplevi);
        Node root = new Node(nodeCounter);
        nodeCounter++;
        if (isAllExamplesPositive() || example.numAttributes() <= 0) {
            Instance newIns = example.get(nodeCounter);
            double lbl = newIns.classValue();
            String strl = newIns.stringValue(newIns.classIndex());
            Label label = new Label(strl, lbl);
        } else {
            Attribute A = findBestAttribute();
            root.setAttribute(A);
            Enumeration<Object> values = A.enumerateValues();
            while(values.hasMoreElements()) {
                Object value = values.nextElement();
                selectSubset(A, value);
                if (this.examplevi.isEmpty()) {
                    Node leafNode = new Node(nodeCounter);
                    nodeCounter++;
                    double mcv = mostCommonValue(example);
                    Label label = new Label(this.data.classAttribute().value((int)mcv), mcv);
                    leafNode.setLabel(label);
                    root.addBranch(Double.valueOf(String.valueOf(value)), leafNode);
                } else {
                    // removeAttributes(A) remove Attributes
                    root.addBranch(Double.valueOf(String.valueOf(value)), makeTree());
                    // addAtributes(A)
                }
            }
        }
        return root;
    }

}
