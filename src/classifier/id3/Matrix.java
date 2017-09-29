package classifier.id3;
// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import java.util.ArrayList;
import java.util.TreeMap;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Iterator;

public class Matrix {
	// Data
	ArrayList< double[] > m_data;

	// Meta-data
	ArrayList< String > m_attr_name;
	ArrayList< TreeMap<String, Integer> > m_str_to_enum;
	ArrayList< TreeMap<Integer, String> > m_enum_to_str;
	
	private ArrayList<Integer> whichRowsToCopy;

	static double MISSING = Double.MAX_VALUE; // representation of missing values in the dataset

	// Creates a 0x0 matrix. You should call loadARFF or setSize next.
	public Matrix() {
	}

	// Copies the specified portion of that matrix into this matrix
	public Matrix(Matrix that, int rowStart, int colStart, int rowCount, int colCount) {
		m_data = new ArrayList<  >();
		for(int j = 0; j < rowCount; j++) {
			double[] rowSrc = that.row(rowStart + j);
			double[] rowDest = new double[colCount];
			for(int i = 0; i < colCount; i++)
				rowDest[i] = rowSrc[colStart + i];
			m_data.add(rowDest);
		}
		m_attr_name = new ArrayList<>();
		m_str_to_enum = new ArrayList<  >();
		m_enum_to_str = new ArrayList<  >();
		for(int i = 0; i < colCount; i++) {
			m_attr_name.add(that.attrName(colStart + i));
			m_str_to_enum.add(that.m_str_to_enum.get(colStart + i));
			m_enum_to_str.add(that.m_enum_to_str.get(colStart + i));
		}
	}
	
	// Custom constructor
	public Matrix (Matrix that, int attribute, int value, int r, int c, ArrayList<Integer> whichToCopy) {
		whichRowsToCopy = new ArrayList<>();
		m_data = new ArrayList<  >();
		if (value == -1) {
			for (int i = 0; i < that.rows(); i++) {
				if (whichToCopy.contains(i)) {
					double[] rowSrc = that.row(i);
					double[] rowDest = new double[c];
					for(int j = 0; j < c; j++)
						rowDest[j] = rowSrc[j];
					m_data.add(rowDest);
				}
			}
		}
		else {
			for(int j = 0; j < that.rows(); j++) {
				if (that.get(j, attribute) == value) {
					double[] rowSrc = that.row(j);
					double[] rowDest = new double[c];
					for(int i = 0; i < c; i++)
						rowDest[i] = rowSrc[i];
					m_data.add(rowDest);
					whichRowsToCopy.add(j);
				}
			}
		}
		m_attr_name = new ArrayList<>();
		m_str_to_enum = new ArrayList<  >();
		m_enum_to_str = new ArrayList<  >();
		for(int i = 0; i < c; i++) {
			m_attr_name.add(that.attrName(i));
			m_str_to_enum.add(that.m_str_to_enum.get(i));
			m_enum_to_str.add(that.m_enum_to_str.get(i));
		}
	}
	
	public ArrayList<Integer> getWhichRowsToCopy () {
		return whichRowsToCopy;
	}

	// Adds a copy of the specified portion of that matrix to this matrix
	public void add(Matrix that, int rowStart, int colStart, int rowCount) throws Exception {
		if(colStart + cols() > that.cols())
			throw new Exception("out of range");
		for(int i = 0; i < cols(); i++) {
			if(that.valueCount(colStart + i) != valueCount(i))
				throw new Exception("incompatible relations");
		}
		for(int j = 0; j < rowCount; j++) {
			double[] rowSrc = that.row(rowStart + j);
			double[] rowDest = new double[cols()];
			for(int i = 0; i < cols(); i++)
				rowDest[i] = rowSrc[colStart + i];
			m_data.add(rowDest);
		}
	}

	// Resizes this matrix (and sets all attributes to be continuous)
	public void setSize(int rows, int cols) {
		m_data = new ArrayList<  >();
		for(int j = 0; j < rows; j++) {
			double[] row = new double[cols];
			m_data.add(row);
		}
		m_attr_name = new ArrayList<>();
		m_str_to_enum = new ArrayList<  >();
		m_enum_to_str = new ArrayList<  >();
		for(int i = 0; i < cols; i++) {
			m_attr_name.add("");
			m_str_to_enum.add(new TreeMap<>());
			m_enum_to_str.add(new TreeMap<>());
		}
	}
	// Returns the number of rows in the matrix
	public int rows() { return m_data.size(); }

	// Returns the number of columns (or attributes) in the matrix
	public int cols() { return m_attr_name.size(); }

	// Returns the specified row
	public double[] row(int r) { return m_data.get(r); }

	// Returns the element at the specified row and column
	public double get(int r, int c) { return m_data.get(r)[c]; }

	// Sets the value at the specified row and column
	public void set(int r, int c, double v) { row(r)[c] = v; }

	// Returns the name of the specified attribute
	public String attrName(int col) { return m_attr_name.get(col); }

	// Set the name of the specified attribute
	public void setAttrName(int col, String name) { m_attr_name.set(col, name); }

	// Returns the name of the specified value
	public String attrValue(int attr, int val) { return m_enum_to_str.get(attr).get(val); }

	// Returns the number of values associated with the specified attribute (or column)
	// 0=continuous, 2=binary, 3=trinary, etc.
	public int valueCount(int col) { return m_enum_to_str.get(col).size(); }

	// Shuffles the row order
	public void shuffle(Random rand) {
		for(int n = rows(); n > 0; n--) {
			int i = rand.nextInt(n);
			double[] tmp = row(n - 1);
			m_data.set(n - 1, row(i));
			m_data.set(i, tmp);
		}
	}

	// Returns the mean of the specified column
	public double columnMean(int col) {
		double sum = 0;
		int count = 0;
		for(int i = 0; i < rows(); i++) {
			double v = get(i, col);
			if(v != MISSING)
			{
				sum += v;
				count++;
			}
		}
		return sum / count;
	}

	// Returns the min value in the specified column
	public double columnMin(int col) {
		double m = MISSING;
		for(int i = 0; i < rows(); i++) {
			double v = get(i, col);
			if(v != MISSING)
			{
				if(m == MISSING || v < m)
					m = v;
			}
		}
		return m;
	}

	// Returns the max value in the specified column
	public double columnMax(int col) {
		double m = MISSING;
		for(int i = 0; i < rows(); i++) {
			double v = get(i, col);
			if(v != MISSING)
			{
				if(m == MISSING || v > m)
					m = v;
			}
		}
		return m;
	}

	// Returns the most common value in the specified column
	public double mostCommonValue(int col) {
		TreeMap<Double, Integer> tm = new TreeMap<>();
		for(int i = 0; i < rows(); i++) {
			double v = get(i, col);
			if(v != MISSING)
			{
				Integer count = tm.get(v);
				if(count == null)
					tm.put(v, 1);
				else
					tm.put(v, count + 1);
			}
		}
		int maxCount = 0;
		double val = MISSING;
		Iterator< Entry<Double, Integer> > it = tm.entrySet().iterator();
		while(it.hasNext())
		{
			Entry<Double, Integer> e = it.next();
			if(e.getValue() > maxCount)
			{
				maxCount = e.getValue();
				val = e.getKey();
			}
		}
		return val;
	}
	
	public double mostCommonValueMissingData(int col, double classification) {
		TreeMap<Double, Integer> tm = new TreeMap<>();
		for(int i = 0; i < rows(); i++) {
			double v = get(i, col);
			if(v != MISSING)
			{
				if (v == classification) {
					Integer count = tm.get(v);
					if(count == null)
						tm.put(v, 1);
					else
						tm.put(v, count + 1);
				}
			}
		}
		int maxCount = 0;
		double val = MISSING;
		Iterator< Entry<Double, Integer> > it = tm.entrySet().iterator();
		while(it.hasNext())
		{
			Entry<Double, Integer> e = it.next();
			if(e.getValue() > maxCount)
			{
				maxCount = e.getValue();
				val = e.getKey();
			}
		}
		return val;
	}

	public void normalize() {
		for(int i = 0; i < cols(); i++) {
			if(valueCount(i) == 0) {
				double min = columnMin(i);
				double max = columnMax(i);
				for(int j = 0; j < rows(); j++) {
					double v = get(j, i);
					if(v != MISSING)
						set(j, i, (v - min) / (max - min));
				}
			}
		}
	}

	public void print() {
		System.out.println("@RELATION Untitled");
		for(int i = 0; i < m_attr_name.size(); i++) {
			System.out.print("@ATTRIBUTE " + m_attr_name.get(i));
			int vals = valueCount(i);
			if(vals == 0)
				System.out.println(" CONTINUOUS");
			else
			{
				System.out.print(" {");
				for(int j = 0; j < vals; j++) {
					if(j > 0)
						System.out.print(", ");
					System.out.print(m_enum_to_str.get(i).get(j));
				}
				System.out.println("}");
			}
		}
		System.out.println("@DATA");
		for(int i = 0; i < rows(); i++) {
			double[] r = row(i);
			for(int j = 0; j < r.length; j++) {
				if(j > 0)
					System.out.print(", ");
				if(valueCount(j) == 0)
					System.out.print(r[j]);
				else
					System.out.print(m_enum_to_str.get(j).get((int)r[j]));
			}
			System.out.println("");
		}
	}
}
