function [ model ] = generic_inverted_pyramid_model(model_index, fov_index, img_size, SAVE_TO_FILE)
%% Single channel variable resolution model
% cell layout at the foveola (assuming 40 overlapping cells of 40" RF radius in all directions)
% center circle
if ~exist("SAVE_TO_FILE","var") || isempty(SAVE_TO_FILE)
    SAVE_TO_FILE = true;
end
if ~exist("img_size","var")
    img_size = [480, 640];
end
if ~exist("fov_index","var")
    fov_index = 1;
end
% FOV options:
%   27deg - for fixating at the center of an image 640x480 pixels
%   54deg - for images of dimension 640x480 pixels with fixation at any image location the diagonal of such image is 800 pixels long, which is about 26.7 deg (assuming about 30 pixels per degree) 
fov_in_deg = [27, 30, 54];

UNFILTERED_FOVEOLA = true; % TRUE to avoid interpolation over neighboring pixels in the highest resolution area (foveola) of the filtered image

[ model ] = extract_model_parameters(model_index, fov_in_deg(fov_index), img_size);

fprintf('[%s] Generating inverted pyramid sample layouts for %d%% pixel budget at FOV of %d degrees ...\n\n',datestr(now), model.sample_percent_name, fov_in_deg(fov_index));
fprintf('[%s] Continuous variable resolution layout (single channel):\n',datestr(now));
fprintf('[%s] --> Foveola region uniform samples (highest resolution).\n',datestr(now));
var_rf_circles_cntr_xy = [0,0];
init_radius = model.var_foveola_cell_radius;
if UNFILTERED_FOVEOLA
    init_radius_factor = 2; % will create a foveola of size equivalent to 40 non-overlapping cells in each dimension
else
    init_radius_factor = 1; % will create a foveola of size equivalent to 40 overlapping cells (or 20 non-overlapping) in each dimension
end    
concentric_radius = init_radius_factor*init_radius;
var_rf_circles_radii = init_radius_factor*init_radius;
for n=2:20
    concentric_radius = (n-1)*init_radius_factor*init_radius;
    num_circles = ceil(360/(2*asind((init_radius/2)/concentric_radius)));
    theta = linspace(0,360,num_circles+1);
    theta = theta(1:end-1);
    x = concentric_radius*cosd(theta);
    y = concentric_radius*sind(theta);
    curr_circ_cntrs = [x(:), y(:)];
    var_rf_circles_cntr_xy = [ var_rf_circles_cntr_xy; curr_circ_cntrs ];
    var_rf_circles_radii = [ var_rf_circles_radii; init_radius*ones(num_circles,1) ];
end

fprintf('[%s] --> Unfiltered foveola samples (non-overlapping):\n',datestr(now));
% override concentric circles for unfiltered version
tmp_var_rf_roi_half_sz = 479;
rnd_cnt = round(var_rf_circles_cntr_xy/model.seconds_per_pixel + tmp_var_rf_roi_half_sz + 1);
tmp_mask = uint8(zeros(2*tmp_var_rf_roi_half_sz+1));
for i=1:size(rnd_cnt,1)
    tmp_mask(rnd_cnt(i,2),rnd_cnt(i,1))=1; 
end
tmp_mask = bwmorph(tmp_mask,'close'); % close blank holes due to rounding
[r,c]=find(tmp_mask);
var_rf_circles_cntr_xy = model.seconds_per_pixel*([r, c] - 1 - tmp_var_rf_roi_half_sz);
var_rf_circles_radii = init_radius*ones(size(var_rf_circles_cntr_xy,1),1);
foveola_end_ind = length(var_rf_circles_radii);


%% cell layout at the fovea outside the foveola. Assuming cells RF is linear with eccenticity (slope 0.1 for the RF diameter, or 0.05 for the RF radius).
fprintf('[%s] --> Fovea and periphery region samples (RF linearly increasing with eccentricity).\n',datestr(now));

odd_circle_fl = true;
while concentric_radius<=((model.fov_in_degrees/2)*3600)
%     radius = 0.05*concentric_radius; % slope 0.1 for the RF diameter, or 0.05 for the RF radius
    radius = init_radius+model.var_ecc_rf_factor*concentric_radius; 
    num_circles = ceil(360/(2*asind((radius/2)/concentric_radius)));
    theta = linspace(0,360,num_circles+1);
    theta = theta(1:end-1);
    theta_diff = mean(diff(theta));   
    if odd_circle_fl 
        odd_circle_fl = false;
    else
        theta = theta + round(theta_diff/2);
        odd_circle_fl = true;
    end
    x = concentric_radius*cosd(theta);
    y = concentric_radius*sind(theta);
    curr_circ_cntrs = [x(:), y(:)];
    var_rf_circles_cntr_xy = [ var_rf_circles_cntr_xy; curr_circ_cntrs ];
    var_rf_circles_radii = [ var_rf_circles_radii; radius*ones(num_circles,1) ];

    n = n + 1;
    concentric_radius = concentric_radius+radius;
end

% visualize variable RF size cells layout
% figure;viscircles(var_rf_circles_cntr_xy, var_rf_circles_radii); grid on; axis equal;

%% Equivalent constant resolution model
% cell layout of the fovea assuming all cells have constant RF size (similar number of cells to the variable RF size layout)
fprintf('[%s] Constant resolution layout (single channel):\n',datestr(now));

fovea_half_sz = max(var_rf_circles_cntr_xy(:,1)); % actual radius of the FOV

const_rf_circles_cntr_xy = [0,0];
init_radius = model.const_rf_radius;
const_rf_circles_radii = model.const_rf_radius;
odd_circle_fl = true;
for nn=2:model.const_half_num_of_uniform_rf_cells
    concentric_radius = (nn-1)*init_radius;
    num_circles = ceil(360/(2*asind((init_radius/2)/concentric_radius)));
    theta = linspace(0,360,num_circles+1);
    theta = theta(1:end-1);
    theta_diff = mean(diff(theta));
    if odd_circle_fl 
        odd_circle_fl = false;
    else
        theta = theta + round(theta_diff/2);
        odd_circle_fl = true;
    end    
    x = concentric_radius*cosd(theta);
    y = concentric_radius*sind(theta);
    curr_circ_cntrs = [x(:), y(:)];
    const_rf_circles_cntr_xy = [ const_rf_circles_cntr_xy; curr_circ_cntrs ];
    const_rf_circles_radii = [ const_rf_circles_radii; init_radius*ones(num_circles,1) ];
end

% visualize constant RF size cells layout
% figure;viscircles(const_rf_circles_cntr_xy, const_rf_circles_radii); grid on; axis equal;

%% RF gaussian filter at each cell location for the single variable resolution model
fprintf('[%s] Generating filters for inverted pyramid layouts...\n\n',datestr(now));
fprintf('[%s] --> Creating filters for the single channel variable resoution model:\n', datestr(now))
tic;

var_rf_roi_half_sz = floor(max(var_rf_circles_cntr_xy(:,1)/model.seconds_per_pixel)+max(var_rf_circles_radii/model.seconds_per_pixel));
gauss_half_sup = 50; % gaussian half support roughly for values larger than 10e-15
var_rf_filter = zeros(2*gauss_half_sup+1,2*gauss_half_sup+1,numel(var_rf_circles_radii),'single');
var_tar_xlim = zeros(numel(var_rf_circles_radii),2,'single');
var_tar_ylim = zeros(numel(var_rf_circles_radii),2,'single');
var_src_xlim = zeros(numel(var_rf_circles_radii),2,'single');
var_src_ylim = zeros(numel(var_rf_circles_radii),2,'single');
for i=1:numel(var_rf_circles_radii)
    c = var_rf_circles_cntr_xy(i,:)/model.seconds_per_pixel;
    [X,Y] = meshgrid(floor(c(1)+0.5)+(-gauss_half_sup:gauss_half_sup),floor(c(2)+0.5)+(-gauss_half_sup:gauss_half_sup));
    rf_gaussian = mvnpdf([X(:) Y(:)],c,eye(2)*(var_rf_circles_radii(i)/model.seconds_per_pixel)^2); % 2D gaussian of each cell RF
    rf_gaussian = reshape(rf_gaussian,1+2*gauss_half_sup,1+2*gauss_half_sup);
    
    var_tar_xlim(i,:) = [max(1, 1+var_rf_roi_half_sz+floor(c(1)-gauss_half_sup +0.5)), min(2*var_rf_roi_half_sz+1, 1+var_rf_roi_half_sz+floor(c(1)+gauss_half_sup +0.5))];
    var_tar_ylim(i,:) = [max(1, 1+var_rf_roi_half_sz+floor(c(2)-gauss_half_sup +0.5)), min(2*var_rf_roi_half_sz+1, 1+var_rf_roi_half_sz+floor(c(2)+gauss_half_sup +0.5))];
    var_src_xlim(i,:) = [max(1, 1-(var_rf_roi_half_sz+floor(c(1)-gauss_half_sup +0.5))), min(2*gauss_half_sup+1, (2*gauss_half_sup+1)-(var_rf_roi_half_sz+1+(floor(c(1)+gauss_half_sup +0.5)-(2*var_rf_roi_half_sz+1))))];
    var_src_ylim(i,:) = [max(1, 1-(var_rf_roi_half_sz+floor(c(2)-gauss_half_sup +0.5))), min(2*gauss_half_sup+1, (2*gauss_half_sup+1)-(var_rf_roi_half_sz+1+(floor(c(2)+gauss_half_sup +0.5)-(2*var_rf_roi_half_sz+1))))];
    var_rf_filter(:,:,i) = single(rf_gaussian);
end
toc;

if SAVE_TO_FILE
    output_filename = sprintf('sampling_models_%d_samp_per_%d_fov_deg_unfilt_foveola.mat', model.sample_percent_name, model.fov_in_degrees);
    fprintf('[%s] --> Saving single-channel variable model filters to file %s.\n', datestr(now), output_filename)
    var_foveola_cell_radius = model.var_foveola_cell_radius;
    seconds_per_pixel = model.seconds_per_pixel;
    var_max_out_img_sz = model.var_max_out_img_sz;
    save(output_filename, '-v7.3', ...
        'var_foveola_cell_radius', 'var_rf_roi_half_sz', 'var_rf_circles_cntr_xy', ...
        'var_rf_circles_radii', 'n',  'var_rf_filter', 'seconds_per_pixel', ...
        'var_max_out_img_sz', 'var_tar_xlim', 'var_tar_ylim', 'var_src_xlim', 'var_src_ylim', 'model');
end

%% RF gaussian filter at each cell location for the equivalent constant resolution model
clear var_rf_filter;
fprintf('[%s] --> Creating filters for the equivalent constant resoution model.\n', datestr(now))
tic;
const_rf_roi_half_sz = floor(max(const_rf_circles_cntr_xy(:,1)/model.seconds_per_pixel)+max(const_rf_circles_radii/model.seconds_per_pixel));
const_rf_filter = zeros(2*gauss_half_sup+1,2*gauss_half_sup+1,numel(const_rf_circles_radii),'single');
const_tar_xlim = zeros(numel(const_rf_circles_radii),2,'single');
const_tar_ylim = zeros(numel(const_rf_circles_radii),2,'single');
const_src_xlim = zeros(numel(const_rf_circles_radii),2,'single');
const_src_ylim = zeros(numel(const_rf_circles_radii),2,'single');
for i=1:numel(const_rf_circles_radii)
    c = const_rf_circles_cntr_xy(i,:)/model.seconds_per_pixel;
    [X,Y] = meshgrid(floor(c(1)+0.5)+(-gauss_half_sup:gauss_half_sup),floor(c(2)+0.5)+(-gauss_half_sup:gauss_half_sup));
    rf_gaussian = mvnpdf([X(:) Y(:)],c,eye(2)*(const_rf_circles_radii(i)/model.seconds_per_pixel)^2); % 2D gaussian of each cell RF
    rf_gaussian = reshape(rf_gaussian,1+2*gauss_half_sup,1+2*gauss_half_sup);
    
    const_tar_xlim(i,:) = [max(1, 1+const_rf_roi_half_sz+floor(c(1)-gauss_half_sup +0.5)), min(2*const_rf_roi_half_sz+1, 1+const_rf_roi_half_sz+floor(c(1)+gauss_half_sup +0.5))];
    const_tar_ylim(i,:) = [max(1, 1+const_rf_roi_half_sz+floor(c(2)-gauss_half_sup +0.5)), min(2*const_rf_roi_half_sz+1, 1+const_rf_roi_half_sz+floor(c(2)+gauss_half_sup +0.5))];
    const_src_xlim(i,:) = [max(1, 1-(const_rf_roi_half_sz+floor(c(1)-gauss_half_sup +0.5))), min(2*gauss_half_sup+1, (2*gauss_half_sup+1)-(const_rf_roi_half_sz+1+(floor(c(1)+gauss_half_sup +0.5)-(2*const_rf_roi_half_sz+1))))];
    const_src_ylim(i,:) = [max(1, 1-(const_rf_roi_half_sz+floor(c(2)-gauss_half_sup +0.5))), min(2*gauss_half_sup+1, (2*gauss_half_sup+1)-(const_rf_roi_half_sz+1+(floor(c(2)+gauss_half_sup +0.5)-(2*const_rf_roi_half_sz+1))))];
    const_rf_filter(:,:,i) = single(rf_gaussian);
end
toc;

if SAVE_TO_FILE
    output_filename = sprintf('sampling_models_%d_samp_per_%d_fov_deg_unfilt_foveola.mat', model.sample_percent_name, model.fov_in_degrees);
    fprintf('[%s] --> Saving equi-constant model filters to file %s.\n', datestr(now), output_filename)
    const_rf_radius = model.const_rf_radius;
    const_max_out_img_sz = model.const_max_out_img_sz;
    const_half_num_of_uniform_rf_cells= model.const_half_num_of_uniform_rf_cells;
    save(output_filename, '-append', ...
        'const_rf_radius', 'const_rf_roi_half_sz', 'const_rf_circles_cntr_xy', ...
        'const_rf_circles_radii', 'const_rf_filter', 'const_half_num_of_uniform_rf_cells', 'const_max_out_img_sz', ...
        'const_tar_xlim', 'const_tar_ylim', 'const_src_xlim', 'const_src_ylim');
end
