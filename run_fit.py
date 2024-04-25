from mantid.simpleapi import *
import numpy as np
import userfunctions

ws_resolution = Load("./BaH2_500C_resolution_1_group.nxs")
ws_to_fit = Load("./BaH2_500C_1_group.nxs")

# Mantid Fit
# minimizer = 'FABADA,ChainLength=10000,StepsBetweenValues=10,ConvergenceCriteria=0.01,InnactiveConvergenceCriterion=5,JumpAcceptanceRate=0.666667,SimAnnealingApplied=1,MaximumTemperature=10,NumRefrigerationSteps=5,SimAnnealingIterations=10000,Overexploration=1,PDF=1,NumberBinsPDF=20'
minimizer = 'Levenberg-Marquardt'


for fit_function in ["Gaussian3D", "Gaussian2D", "DoubleWell"]:

    function_string = f"""
    composite=Convolution,FixResolution=true,NumDeriv=true;
    name=Resolution,Workspace={ws_resolution.name()},WorkspaceIndex=0;
    name={fit_function}
    """

    Fit(
        Function=function_string,
        InputWorkspace=ws_to_fit,
        IgnoreInvalidData=True,
        MaxIterations=20000,
        Output=ws_to_fit.name() + "_Fit" + fit_function,
        Minimizer=minimizer
        )
