function [ model ] = extract_model_parameters(model_index, fov_in_deg, img_size)
if ~exist("model_index","var")
    model_index = 1;
end
if ~exist("fov_in_deg","var")
    fov_in_deg = 54;
    % fov_in_degrees = 27; % for fixating at the center of an image 640x480 pixels
end
if ~exist("img_size","var")
    img_size = [480, 640];
end
model.img_size = img_size;
model.fullres_samples = prod(img_size);
model.fov_in_degrees = fov_in_deg;

% single channel variable resolution model parameters (following Poggio et al. - Poggio, T., Mutch, J., & Isik, L. (2014). Computational role of eccentricity dependent cortical magnification (Issue CBMM Memo 017). http://arxiv.org/abs/1406.1770
% model's outline:
%   *   the foveola positioned at the center of fixation provides the finest resolution with 40 ganglion cells spanning 26' (minutes of arc), each with RF (receptive field) spanning 1'20" in diameter (equivalent to a single photoreceptor)
%   *   in the fovea, the retinal area surrounding the foveola, the RF of the ganglion cells increase linearly with eccentricity (distance from the center of fixation) at a rate of 0.1.
%   *   the fovea (positioned at the center of the macula) provides high resolution and spans about 6 visual degrees.
%   *   there are about 40 ganglion cells at the boundaries of the fovea, each with RF spanning 42' in diameter.

% in the model, the finest resolution ganglion cell has a RF spanning 1'20"
% however, the finest resolution of a digital screen is 1 pixel. 
% in most screens, 1 pixel spans 1'53" at a viewing distance of 0.5 meters.
% this is derived from a pixel density of 96 ppi (pixels per inch) and HD
% resolution of 1920x1080 pixels for 23.8 inch screens
% in this implemetation we use 2' (120") per pixel (or cell) for simplicity
model.seconds_per_pixel = 120; % in seconds of arc - we assume that 1 pixel is equivalent to a foveola cell (the highest image resolution)

model.var_foveola_cell_radius = model.seconds_per_pixel / 2; % in seconds of arc (in the original model this is 40" RF radius)

sample_percent_model_names = [3, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90];
model.sample_percent_name = sample_percent_model_names(model_index);

% below are empirical values for the eccentricity factor for computing the RF radius as: RF_radius = ecc_factor * eccentricity
% (1) var_ecc_rf_factor = 0.05; % provides 2.4% (7292 samples) at 27deg, 2.5% (7538 samples) at 30deg, or 2.9% (9027 samples) at 54deg of the full-resolution
% (2) var_ecc_rf_factor = 0.0195; % provides 10% (30596 samples) at 27deg, 10.5% (32118 samples) at 30deg, or 13.4% (41091 samples) at 54deg of the full-resolution
% (3) var_ecc_rf_factor = 0.015; % provides 14.7% (45267 samples) at 27deg, 15.6% (47990 samples) at 30deg, or 20.4% (62688 samples) at 54deg of the full-resolution
% (4) var_ecc_rf_factor = 0.0122; % provides 20% (61479 samples) at 27deg, 21.2% (65243 samples) at 30deg, or 28.3% (86952 samples) at 54deg of the full-resolution
% (5) var_ecc_rf_factor = 0.0092; % provides 29.9% (91867 samples) at 27deg, 31.9% (97930 samples) at 30deg, or 44.3% (136021 samples) at 54deg of the full-resolution )
% (6) var_ecc_rf_factor = 0.0074; % provides 40.3% (123660 samples) at 27deg, 43.1% (132464 samples) at 30deg, or 61% (187440 samples) at 54deg of the full-resolution )
% (7) var_ecc_rf_factor = 0.0063; % provides 49.5% (152192 samples) at 27deg, 53.4% (163970 samples) at 30deg, or 77.2% (237089 samples) at 54deg of the full-resolution )
% (8) var_ecc_rf_factor = 0.0054; % provides 60.3% (185280 samples) at 27deg, 65.3% (200584 samples) at 30deg,  or 96.5% (296390 samples) at 54deg of the full-resolution )
% (9) var_ecc_rf_factor = 0.0048; % provides 69.8% (214268 samples) at 27deg, 75.9% (233223 samples) at 30deg,  or 113.7% (349292 samples) at 54deg of the full-resolution )
% (10) var_ecc_rf_factor = 0.0043; % provides 79.5% (244219 samples) at 27deg, 86.6% (266048 samples) at 30deg, or 132.5% (406890 samples) at 54deg of the full-resolution )
% (11) var_ecc_rf_factor = 0.00385; % provides 90.1% (276859 samples) at 27deg, 98.7% (303137 samples) at 30deg, or 153.7% (472279 samples) at 54deg of the full-resolution )
if model.fov_in_degrees == 27
    sample_ratios = [7292, 30596, 45267, 61479, 91867, 123660, 152192, 185280, 214268, 244219, 276859]./model.fullres_samples;
else
    if model.fov_in_degrees == 30
        sample_ratios = [7538, 32118, 47990, 65243, 97930, 132464, 163970, 200584, 233223, 266048, 303137]./model.fullres_samples;
    else
        samle_ratios = [9027, 41091, 62688, 86952, 136021, 187440, 237089, 296390, 349292, 406890, 472279]./model.fullres_samples;   
    end
end
ecc_rf_factors = [0.05, 0.0195, 0.0150, 0.0122, 0.0092, 0.0074, 0.0063, 0.0054, 0.0048, 0.0043, 0.00385];
var_est_splinefunc = spline(sample_ratios, ecc_rf_factors); % calculate a spline function as a predictor for the eccentricity RF radius factor

model.sample_ratio = sample_ratios(model_index);

model.var_ecc_rf_factor = fnval(var_est_splinefunc, model.sample_ratio); % increase factor of cell RF radius with eccentricity

model.var_max_out_img_sz = 1600;  % 1600x1600


% the uniform (equi-constant) sampling model's parameters, with
% equivalent number of samples to the single variable sampling model above
% (1) const_rf_radius = 1010; const_half_num_of_uniform_rf_cells = 48, 54 or 96; % (27, 30 or 54 deg)
%     Number of cells (FOV 27d): 7111 % Number of cells (FOV 30d): 9018 % Number of cells (FOV 54d): 28698 
%     8.42 pixels spacing between two samples (cells) this spacing was defined to match the number of cells of the variable resolution model with FOV radius of 13.5 deg: 7111 cells constant ~= 7292 cells variable at eccentricity of 13.5deg
% (2) const_rf_radius = 490; const_half_num_of_uniform_rf_cells = 99, 111 or 198; % (27, 30 or 54 deg)
%     Number of cells (FOV 27d): 30528 % Number of cells (FOV 30d): 38414 % Number of cells (FOV 54d): 122638 
%     4.08 pixels spacing between two samples (cells) this spacing was defined to match the number of cells of the variable resolution model with FOV radius of 13.5 deg: 30528 cells constant ~= 30596 cells variable at eccentricity of 13.5deg
% (3) const_rf_radius = 400; const_half_num_of_uniform_rf_cells = 121, 135 or 243; % (27, 30 or 54 deg)
%     Number of cells (FOV 27d): 45675 % Number of cells (FOV 30d): 56897 % Number of cells (FOV 54d): 184864 
%     3.33 pixels spacing between two samples (cells) this spacing was defined to match the number of cells of the variable resolution model with FOV radius of 13.5 deg: 45675 cells constant ~= 45267 cells variable at eccentricity of 13.5deg
% (4) const_rf_radius = 345; const_half_num_of_uniform_rf_cells = 140, 157 or 282; % (27, 30 or 54 deg)
%     Number of cells (FOV 27d): 61204 % Number of cells (FOV 30d): 77021  Number of cells (FOV 54d): 249086 
%     2.88 pixels spacing between two samples (cells) this spacing was defined to match the number of cells of the variable resolution model with FOV radius of 13.5 deg: 61204 cells constant ~= 61479 cells variable at eccentricity of 13.5deg
% (5) const_rf_radius = 280; const_half_num_of_uniform_rf_cells = 171, 193 or 347; % (27, 30 or 54 deg)
%     Number of cells (FOV 27d): 91411 % Number of cells (FOV 30d): 116510 % Number of cells (FOV 54d): 377357 
%     2.33 pixels spacing between two samples (cells) this spacing was defined to match the number of cells of the variable resolution model with FOV radius of 13.5 deg: 91411 cells constant ~= 91867 cells variable at eccentricity of 13.5deg
% (6) const_rf_radius = 245; const_half_num_of_uniform_rf_cells = 199, 221 or 397; % (27, 30 or 54 deg)
%     Number of cells (FOV 27d): 123883 % Number of cells (FOV 30d): 152854 % Number of cells (FOV 54d): 494093 
%     2.04 pixels spacing between two samples (cells) this spacing was defined to match the number of cells of the variable resolution model with FOV radius of 13.5 deg: 123883 cells constant ~= 123660 cells variable at eccentricity of 13.5deg
% (7) const_rf_radius = 219; const_half_num_of_uniform_rf_cells = 221, 247 or 444; % (27, 30 or 54 deg)
%     Number of cells (FOV 27d): 152854  % Number of cells (FOV 30d): 191011 % Number of cells (FOV 54d): 618146 
%     1.83 pixels spacing between two samples (cells) this spacing was defined to match the number of cells of the variable resolution model with FOV radius of 13.5 deg: 152854 cells constant ~= 152192 cells variable at eccentricity of 13.5deg
% (8) const_rf_radius = 201; const_half_num_of_uniform_rf_cells = 243, 269 or 484; % (27, 30 or 54 deg) 
%     Number of cells (FOV 27d): 184864 % Number of cells (FOV 30d): 226616 % Number of cells (FOV 54d): 734655 
%     1.68 pixels spacing between two samples (cells) this spacing was defined to match the number of cells of the variable resolution model with FOV radius of 13.5 deg: 184864 cells constant ~= 185280 cells variable at eccentricity of 13.5deg
% (9) const_rf_radius = 185; const_half_num_of_uniform_rf_cells = 262, 292 or 526; % (27, 30 or 54 deg)
%     Number of cells (FOV 27d): 214957 % Number of cells (FOV 30d): 267092 % Number of cells (FOV 54d): 867811 
%     1.54 pixels spacing between two samples (cells) this spacing was defined to match the number of cells of the variable resolution model with FOV radius of 13.5 deg: 214957 cells constant ~= 214268 cells variable at eccentricity of 13.5deg
% (10) const_rf_radius = 173; const_half_num_of_uniform_rf_cells = 279, 313 or 562; % (27, 30 or 54 deg)
%      Number of cells (FOV 27d): 243806 % Number of cells (FOV 30d): 306950 % Number of cells (FOV 54d): 990766 
%      1.44 pixels spacing between two samples (cells) this spacing was defined to match the number of cells of the variable resolution model with FOV radius of 13.5 deg: 243806 cells constant ~= 244219 cells variable at eccentricity of 13.5deg
% (11) const_rf_radius = 165; const_half_num_of_uniform_rf_cells = 297, 328 or 589; % (27, 30 or 54 deg)
%      Number of cells (FOV 27d): 276330 % Number of cells (FOV 30d): 337117 % Number of cells (FOV 54d): 1088325 
%      1.38 pixels spacing between two samples (cells) this spacing was defined to match the number of cells of the variableresolution model with FOV radius of 13.5 deg: 276330 cells constant ~= 276859 cells variable at eccentricity of 13.5deg

const_rf_radius = [1010, 490, 400, 345, 280, 245, 219, 201, 185, 173, 165];
const_rf_radius_est_splinefunc = spline(sample_ratios, const_rf_radius); % calculate a spline function as a predictor for the uniform RF radius
model.const_rf_radius = fnval(const_rf_radius_est_splinefunc, model.sample_ratio);
if model.fov_in_degrees == 27
    const_half_num_cells = [48, 99, 121, 140, 171, 199, 221, 243, 262, 279, 297];
else
    if model.fov_in_degrees == 30
        const_half_num_cells = [54, 111, 135, 157, 193, 221, 247, 269, 292, 313, 328];
    else
        const_half_num_cells = [96, 198, 243, 282, 347, 397, 444, 484, 526, 562, 589];
    end
end
const_rf_half_num_cells_est_splinefunc = spline(sample_ratios, const_half_num_cells); % calculate a spline function as a predictor for the uniform half number of cells along the FOV diameter
model.const_half_num_of_uniform_rf_cells = fnval(const_rf_half_num_cells_est_splinefunc, model.sample_ratio);

model.const_max_out_img_sz = 1600; % 1600x1600


