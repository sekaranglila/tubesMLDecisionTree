package classifier.id3;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.LinkedHashMap;
import weka.core.Attribute;

public class Node {
	
	private Label _label;
	private LinkedHashMap<Double, Node> _branches;
	private Attribute _attribute;
	private int _nodeID;
	
	public Node (int nodeCounter) {
		_label = new Label();
		_branches = new LinkedHashMap<>();
		_attribute = new Attribute("");
		_nodeID = nodeCounter;
	}
	
	public Node (Node that) {
		this(-1);
	}
	
	public int getNodeID() {
		return _nodeID;
	}
	
	public Label getLabel() {
		return _label;
	}
	
	public void setLabel(Label label) {
		_label = label;
	}

	public void setAttribute(Attribute bestAttribute) {
		_attribute = bestAttribute;
	}
	
	public void addBranch(double value, Node node) {
		_branches.put(value, node);
	}
	
	public LinkedHashMap<Double, Node> getBranches() {
		return _branches;
	}
	
	public Attribute getAttribute() {
		return _attribute;
	}
	
        private void dumpDot(PrintWriter out, Matrix examples, Matrix targetAttributes) {
		String myLabel = "";
		if (_branches.isEmpty()) {
			myLabel = _label.getStrValue();
		}
		else {
			myLabel = _attribute.name();
		}
		out.println("  " + _nodeID + " [label=\"" + myLabel + "\"];");//" + toString() + "\"];\n");
		
		if (!_branches.isEmpty()) {
			for (double key : _branches.keySet()) {
				Node childNode = _branches.get(key);
				String edgeLabel = "";
				edgeLabel = examples.attrValue(_attribute.index(), (int)key);
				int childNodeID = childNode.getNodeID();
				out.print("  " + _nodeID + " -> " + childNodeID);
				out.print(" [label=\" " + edgeLabel + "\"];\n");
//				System.out.println(edgeLabel);
				childNode.dumpDot(out, examples, targetAttributes);
			}
		}
	}
        
	public void dumpDot(Matrix examples, Matrix targetAttributes, String fileName) throws IOException {
		PrintWriter out = new PrintWriter(new File(fileName));
		out.println("digraph DecisionTree {");
		out.println("graph [ordering=\"out\"];");
		dumpDot(out, examples, targetAttributes);
		out.println("}");
		out.close();
	}
	
	public double makeDecision (double[] features, int attribute) {
		double decision = 0;
		if (_branches.isEmpty())
			return _label.getValue();
		else {
			for (double branchValue : _branches.keySet()) {
				double value = features[attribute];
				if (branchValue == value) {
					Node childNode = _branches.get(branchValue);
					decision = childNode.makeDecision(features, childNode.getAttribute().index());
				}
			}
		}
		return decision;
	}
}
