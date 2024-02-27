import pandas as pd
import argparse
import os
import uproot
import ROOT
from tqdm import tqdm
import awkward as ak
import numpy as np
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from numba import jit, prange

parser = argparse.ArgumentParser(description='Analyse the stability froma  certain run',epilog='Text at the bottom of help')
parser.add_argument('-csv','--csv',help='csv with matched runnumber and TPH', action='store', type=str,default='./Feb24Out.csv')
parser.add_argument('-i','--input',help='Path of folder with the reco files', action='store', type=str,default="./Feb2024-MANGOlino-55Fe")
parser.add_argument('-f','--fit',help='proceed with fitting the LY as function of the TP', action='store', type=int,default=None)
parser.add_argument('-a','--astep', nargs=2,help='Range for a should be a list of 2 elements', action='store', type=float,default=[-2,4])
parser.add_argument('-b','--bstep', nargs=2,help='range for b should be a list of 2 elements', action='store', type=float,default=[-13,-8])
parser.add_argument('-j','--cores',help='set numebr of cores', action='store', type=int,default=8)
args = parser.parse_args()

def nparr(string):
    return np.array(string, dtype="d")
def get_files_in_folder(folder_path):
    """
    Returns a list of all files in the specified folder.
    Parameters:
    - folder_path: Path to the folder from which to list files.
    Returns:
    - List of file names (with paths) in the specified folder.
    """
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return files
def get_sc_integral(file,cuts=None):
    try:
        events=uproot.open(file+":Events")
    except:
        print("Failed to open (maybe empty)",file)
    if cuts is None: cutIntegral=events.arrays(["sc_integral"])
    else: cutIntegral=events.arrays(["sc_integral"],f"(sc_integral>{cuts[0]}) & (sc_integral<{cuts[1]})")
    ScIntegralTemp = [next(iter(d.values())) for d in ak.to_list(cutIntegral)]
    ScIntegral = [item for sublist in ScIntegralTemp for item in sublist]
    return ScIntegral
def fill_h(histo_name, array):
    for x in range (len(array)):
        histo_name.Fill((np.array(array[x] ,dtype="d")))
def hist(list, x_name, channels=100, linecolor=4, linewidth=4,write=True):
    array=np.array(list ,dtype="d")
    hist=ROOT.TH1D(x_name,x_name,channels,0.99*np.min(array),1.01*np.max(array))
    fill_h(hist,array)
    hist.SetLineColor(linecolor)
    hist.SetLineWidth(linewidth)
    hist.GetXaxis().SetTitle(x_name)
    hist.GetYaxis().SetTitle("Entries")
    if write==True: hist.Write()
    #hist.SetStats(False)
    hist.GetYaxis().SetMaxDigits(3);
    hist.GetXaxis().SetMaxDigits(3);
    return hist
def create_fill_TH2(name,x_name,y_name,z_name, x_vals, y_vals, weights, x_bins=15, y_bins=15,write=True):
    hist = ROOT.TH2F(name, name, x_bins, 0.99*np.min(x_vals), 1.01*np.max(x_vals), y_bins, 0.99*np.min(y_vals), 1.01*np.max(y_vals))
    # Fill the histogram with the data
    for x, y, weight in zip(x_vals, y_vals, weights):
        hist.Fill(x, y, weight)
    # Set axis titles
    hist.GetXaxis().SetTitle(x_name)
    hist.GetYaxis().SetTitle(y_name)
    hist.GetZaxis().SetTitle(z_name)  # Set the z-axis title
    # Draw the histogram
    hist.Draw("COLZ")  # Use the "COLZ" option to draw with a color palette
    if write==True: hist.Write()
    return hist
def grapherr(x,y,ex,ey,x_string, y_string,name=None, color=4, markerstyle=22, markersize=2,write=True):
        plot = ROOT.TGraphErrors(len(x),  np.array(x  ,dtype="d")  ,   np.array(y  ,dtype="d") , np.array(   ex   ,dtype="d"),np.array( ey   ,dtype="d"))
        if name is None: plot.SetNameTitle(y_string+" vs "+x_string,y_string+" vs "+x_string)
        else: plot.SetNameTitle(name, name)
        plot.GetXaxis().SetTitle(x_string)
        plot.GetYaxis().SetTitle(y_string)
        plot.SetMarkerColor(color)#blue
        plot.SetMarkerStyle(markerstyle)
        plot.SetMarkerSize(markersize)
        if write==True: plot.Write()
        return plot
def graph(x,y,x_string, y_string,name=None, color=4, markerstyle=22, markersize=2,write=True):
        plot = ROOT.TGraphErrors(len(x),  np.array(x  ,dtype="d")  ,   np.array(y  ,dtype="d"))
        if name is None: plot.SetNameTitle(y_string+" vs "+x_string,y_string+" vs "+x_string)
        else: plot.SetNameTitle(name, name)
        plot.GetXaxis().SetTitle(x_string)
        plot.GetYaxis().SetTitle(y_string)
        plot.SetMarkerColor(color)#blue
        plot.SetMarkerStyle(markerstyle)
        plot.SetMarkerSize(markersize)
        if write==True: plot.Write()
        return plot
def plot_tgraph2d(x, y, z, title="3D Plot", x_title="X axis", y_title="Y axis", z_title="Z axis",write=True, color=4, markerstyle=22, markersize=2,plot=False):
    """
    Create and draw a TGraph2D plot.

    Parameters:
    - x, y, z: Arrays of x, y, and z coordinates of the points.
    - title: Title of the plot.
    - x_title, y_title, z_title: Titles for the X, Y, and Z axes.
    - draw_option: Drawing option as a string (e.g., "P" for points, "TRI" for triangles, etc.).

    Returns:
    - The TGraph2D object.
    """
    # Convert input data to numpy arrays if they aren't already
    x_array = np.array(x, dtype="float64")
    y_array = np.array(y, dtype="float64")
    z_array = np.array(z, dtype="float64")
    # Create the TGraph2D object
    graph = ROOT.TGraph2D(len(x_array), x_array, y_array, z_array)
    # Set titles
    graph.SetNameTitle(title,title)
    graph.GetXaxis().SetTitle(x_title)
    graph.GetYaxis().SetTitle(y_title)
    graph.GetZaxis().SetTitle(z_title)
    graph.SetMarkerColor(color)#blue
    graph.SetMarkerStyle(markerstyle)
    graph.SetMarkerSize(markersize)
    # Draw the graph
    graph.Draw("COLZ")
    if write==True: graph.Write()
    if plot==True:
        can1=ROOT.TCanvas("Chi2 values","Chi2 values", 1000, 1000)
        can1.SetFillColor(0);
        can1.SetBorderMode(0);
        can1.SetBorderSize(2);
        can1.SetLeftMargin(0.15);
        can1.SetRightMargin(0.2);
        can1.SetTopMargin(0.1);
        can1.SetBottomMargin(0.1);
        can1.SetFrameBorderMode(0);
        can1.SetFrameBorderMode(0);
        can1.SetFixedAspectRatio();

        graph.Draw("colz")
        can1.Update()
        can1.SaveAs("./chi2.png")

    return graph

df_in=pd.read_csv(args.csv)
files=get_files_in_folder(args.input)

main=ROOT.TFile("anal.root","RECREATE")#root file creation
main.mkdir("HISTs")
main.mkdir("varTIME")
main.mkdir("varHIST")
main.mkdir("varCORR")
main.mkdir("corrLY")

gaus=ROOT.TF1("gaus", "gaus(0)",5000,30000)
gauspol=ROOT.TF1("gauspol", "gaus(0)+pol1(3)",5000,30000)

#FIT sc_integrals
main.cd("HISTs")
LY,errLY=[],[]
print("FITTING SC_INTEGRALS...")
for run in tqdm(df_in["Run"]):
    #print(run)
    file=f"{args.input}/reco_run{run}_3D.root"
    sc_temp=get_sc_integral(file,cuts=[0,50000])
    hTemp=hist(sc_temp,f"ScInt run{run}",write=False)
    hTemp.Fit("gaus","RQ","r")
    gauspol.SetParameters(gaus.GetParameter(0),gaus.GetParameter(1),gaus.GetParameter(2))
    hTemp.Fit("gauspol","RQ","r")
    hTemp.Write()
    LY.append(gauspol.GetParameter(1))
    errLY.append(gauspol.GetParError(1))

# Ensure the length of the 'runs' list matches the number of rows in 'nearest_rows' DataFrame
if len(df_in) == len(LY):
    df_in['LY'] = LY
    df_in['errLY'] = errLY
else:
    print("The lengths do not match. Please check your data.")

#print(df_in)

# PLOT variables histograms and against time
run_data = df_in['Run'].values  # Extracting the 'Run' column as numpy array
# Iterate over the columns you want to plot against 'Run'
for column in df_in.columns[2:]:  # Skipping the first two columns ('Run' and 'Time')
    y_data = df_in[column].values  # Extracting the current column's data as numpy array
    x_string = 'Run'
    y_string = column
    # Call your graph function
    main.cd("varTIME")
    plot = graph(run_data, y_data, x_string, y_string, name=column, color=4, markerstyle=22, markersize=2)
    # Assuming you have a TCanvas or similar to draw or you handle the drawing inside your graph function
    main.cd("varHIST")
    temphist = hist(y_data,y_string,channels=20)

main.cd("varCORR")
# Variables to consider for plotting
variables = ['Temperature', 'Pressure', 'Humidity', 'VOC']
# Generate all combinations of the variables, two at a time
combinations = list(itertools.combinations(variables, 2))
# df_in is your input DataFrame
for combination in combinations:
    # Extracting data for the two variables in the current combination
    x_data = df_in[combination[0]].values
    y_data = df_in[combination[1]].values

    # Since your function does not handle errors in x and y, 
    # we will not pass error arrays and assume no error handling is needed here
    x_string = combination[0]
    y_string = combination[1]

    # Call your graph function for each pair
    # Adjust color, markerstyle, markersize as needed
    plot = graph(x_data, y_data, x_string, y_string, name=f"{y_string} vs {x_string}", color=4, markerstyle=22, markersize=2)
    # Handle plot drawing or saving within the ROOT environment as necessary


#plot correlation of LY
main.cd()
# Assuming `df` is your DataFrame
LY = df_in['LY'].values
errLY = df_in['errLY'].values
# Variables to plot against LY
variables = ['Temperature', 'Pressure', 'Humidity', 'VOC']
# Iterate over the variables
for var in variables:
    x = df_in[var].values  # The variable values
    y = LY  # LY values
    ex = np.zeros(len(x))  # Assuming no error in the x-direction
    ey = errLY  # errLY values as the y-direction error

    x_string = var
    y_string = 'LY'

    # Call your grapherr function
    plot = grapherr(x, y, ex, ey, x_string, y_string, name=y_string+" vs "+x_string, color=4, markerstyle=22, markersize=2)
    # Assuming you have a TCanvas or similar to draw or you handle the drawing inside your grapherr function
create_fill_TH2("LY vs T and P","P","T","LY", df_in["Pressure"], df_in["Temperature"], df_in["LY"], x_bins=15, y_bins=15,write=True)
plot_tgraph2d(df_in["Pressure"], df_in["Temperature"], df_in["LY"],"LY vs T and P","P","T","LY")


print("FITTING LY(T,P)...")
#TP fitting
if args.fit is not None:
    # Basic data extraction and preparation, using float64 for precision unless float64 is sufficient.
    # Convert data types for efficiency
    LY = np.array(df_in["LY"], dtype=np.float64)
    errLY = np.array(df_in["errLY"], dtype=np.float64)
    P = np.array(df_in["Pressure"], dtype=np.float64)
    T = np.array(df_in["Temperature"], dtype=np.float64)

    LY0 = LY[0]
    errLY0 = errLY[0]
    P0 = P[0]
    T0 = T[0]

    # Normalize LY
    LY_norm = LY / LY0
    errLY_norm = np.sqrt(LY_norm * ((errLY / LY) ** 2 + (errLY0 / LY0) ** 2))

    # Adjusting the chi2func for Numba optimization
    def chi2func(LY, dLY, a, b, p, t, p0, t0, sigma_p, sigma_t):
        #model = (1 / np.exp(1)) * np.exp(((p0 / p) ** a) * ((t / t0) ** b))
        model = ((p0 / p) ** a) * ((t / t0) ** b)

        if sigma_p == 0 and sigma_t == 0:
            total_dLY_squared = dLY**2  # Use dLY squared directly if there's no additional error.
        else:
            partial_Y_p = -a * (p0 / p)**a * (t / t0)**b * model / p
            partial_Y_t = b * (p0 / p)**a * (t / t0)**b * model / t
            sigma_additional = np.sqrt((partial_Y_p * sigma_p)**2 + (partial_Y_t * sigma_t)**2)
            total_dLY_squared = dLY**2 + sigma_additional**2  # Adjusted error squared

        chiq = np.sum(np.square(LY - model) / total_dLY_squared)
        return chiq / (len(LY) - 2)

    steps = 100000
    a_values = np.random.uniform(args.astep[0], args.astep[1], steps)
    b_values = np.random.uniform(args.bstep[0], args.bstep[1], steps)

    #This additional sigma take into consideration the errors on P and T measured only
    chi2 = np.empty(steps, dtype=np.float64)
    sigmaP,sigmaT=0,0#Pa and K
    for j in tqdm(range(steps)):
        chi2[j] = chi2func(LY_norm, errLY_norm, a_values[j], b_values[j], P, T, P0, T0,sigmaP,sigmaT)

    # Create a mask for elements in chi2 that are <= 10
    mask = chi2 <= 100
    # Apply the mask to all arrays
    chi2 = chi2[mask]
    a_values = a_values[mask]
    b_values = b_values[mask]
    plot_tgraph2d(a_values, b_values, chi2, "chi2 vs a and b","a","b","CHI2",plot=True)

    # Finding the minimum chi2 value
    min_index = np.argmin(chi2)
    a_min, b_min = a_values[min_index], b_values[min_index]
    print(f"The minimum chi2 value is: {chi2[min_index]}, corresponding to a = {a_min} and b = {b_min}.")
    #plot_tgraph2d(P,T,  (1/np.exp(1))*np.exp(((P0/P)**a_min)*((T/T0)**b_min))  ,"modelLY vs T and P","P","T","modelLY")
    #plot_tgraph2d(a_values, b_values, chi2, "chi2 vs a and b","a","b","CHI2")

    ##PLOT corrected LY
    def correction(LY_norm,P,T,P0,T0,a,b):
        return LY_norm/((1 / np.exp(1)) * np.exp(((P0 / P) ** a) * ((T / T0) ** b)))
    LY_corr=correction(LY_norm,P,T,P0,T0,a_min,b_min)
    main.cd("corrLY")
    hist(LY_corr,"LY_corr",channels=20)
    hist(LY_norm,"LY_norm",channels=20)

    plot = graph(P, LY_corr, "Pressure(Pa)", "LY_corr")
    plot = graph(P, LY_norm, "Pressure(Pa)", "LY_norm", color=2, markerstyle=23)
    plot = graph(T, LY_corr, "Temperature(K)", "LY_corr")
    plot = graph(T, LY_norm, "Temperature(K)", "LY_norm", color=2, markerstyle=23)
    plot = graph(df_in["Run"], LY_corr, "Run Num", "LY_corr")
    plot = graph(df_in["Run"], LY_norm, "Run Num", "LY_norm", color=2, markerstyle=23)
