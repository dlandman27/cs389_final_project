\documentclass{article}

\usepackage[english]{babel}
\usepackage[letterpaper,top=2cm,bottom=2cm,left=3cm,right=3cm,marginparwidth=1.75cm]{geometry}

% Useful packages
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage[colorlinks=true, allcolors=blue]{hyperref}

\title{Predicting Prices of Cars}
\author{Dylan, Maanas, Tirth}

\begin{document}
\maketitle

\section{Project Description}

\subsection{Motivation} 

Our group was motivated to design a project that would not only be cool, but could also provide real world applications. We know that ML is used so widely in the automobile industry, particularly in recommending cars for potential buyers, enhancing navigation systems, and even making self-driving cars safer. 
\newline
\newline
There are many features that make cars unique, such as make, model, shape, year made, drive terrain, price, and much more. This gives us plenty of data to work with, allowing us to extract these details and classify new cars by their price. For instance, a potential buyer is having trouble estimating the price of a car online, and wants to compare the price of two cars through a dealership vs through previous owners. There is a large discrepancy on the price of a car, one could get two completely different estimates when assessing the real value of a car because of depreciation, damages, and many more factors. 
\newline
\newline
This is where we wanted to provide help with that.

\subsection{Architecture}
We will be designing a Convolutional Neural Network(CNN) to determine the price of a car based on the brand, price, year made, make & model, and a photograph of the car (from any angle). Our model will consist of 3 convolutional layers and 3 linear layers, and we will implement ReLu() and MaxPool2D() functions. We will use the MSE() gradient descent loss function to train the model, using various hyperparameter searches to determine the best line of the model. Furthermore, we will process our data beforehand and make sure that all of our images are the same dimensions by cropping out unnecessary background noise.   

\subsection{Dataset}
We are using the The Car Connection Picture Dataset. According to this source, it "scrapes around 297,000 pictures, of which around 198,000 unique URLs." At the very end we should have approximately 60,000 pictures of relevant cars from the 2000-2020 era. However we do have to filter some pictures of car interiors, since it provides no intrinsic value to our model. Each of these images are labeled by their feature names: 
\newline
'Make', 'Model', 'Year', 'MSRP', 'Front Wheel Size (in)', 'SAE Net Horsepower @ RPM',
'Displacement', 'Engine Type', 'Width, Max w/o mirrors (in)', 'Height, Overall (in)',
'Length, Overall (in)', 'Gas Mileage', 'Drivetrain', 'Passenger Capacity', 'Passenger Doors',
'Body Style'

\section{Literature Review} 

\subsection{Links}

 Review 1: \href{https://paperswithcode.com/paper/ai-blue-book-vehicle-price-prediction-using}{AI Blue Book: Vehicle Price Prediction using Visual Features} 
 \newline
  \newline
 Summary: This paper looks at several ML models in which could be used to perform linear regressions and stochastic gradient descent to determine the price of a car based on an image. Yang et al used a combination of several deep Convolutional Neural Networks (CNN) to classify the datasets of images and determine the price. They determined that using CNN's provided the most accurate results of all the methods provided. For the regression processing, instead of using lines, Yang et al used histograms of oriented gradients (HOG) to represent the linear regression of the model. To perform the convolutional layers, the team developed a new architecture called PriceNet. This architecture replaces 3x3 convolutions used in a similar architectures (SqueezeNet) with fire modules. Fire modules work by creating first a convolutional 1x1 layer and then sampled by combinations of both 1x1 and 3x3 convolutional layers. They concoluded that their network outperformed other models signifigantly and were able to better visualize the price, based on visual features of a car. With this technology, and with more tweaking, companies could determine the parts of cars that cause the price to be more and use better photos to market their products.
 \newline 
  
 Review 2: \href{https://paperswithcode.com/paper/assessing-car-damage-using-mask-r-cnn}{Assessing Car Damage using Mask R-CNN}
 \newline
  \newline
  
  
 Review 3: \href{https://paperswithcode.com/paper/price-aware-recommendation-with-graph}{Price-aware Recommendation with Graph Convolutional Networks} 
 Summary: Recommendation systems are extremely important when it comes to targeted marketing of customers. And an important factor if a recommendation will be bought by a potential buyer will largely depend on the price of the item at hand. However, the author suggests a term, WTP, where it makes sense for the user to be willing to pay (WTP) a price for the item at interest. Purchase history of a user and the variable tolerances of price in different topics play are two examples the author identifies to understand the boundaries of WTP. Giving larger weightage to price points, the author proposes a solution that implements Graph Convolution Networks, and the factors of items and prices take place. This helps analyze the user's category-dependent price awareness. We felt this was largely relevant to our project due to it's emphasis on price optimization and thought that their unique solution of PUP Graph Convolutional Networks would be applicable as well. 
 \newline

\section{Preliminary} 

\subsection{Graph #1}

\subsection{Graph #2}

\end{document}
