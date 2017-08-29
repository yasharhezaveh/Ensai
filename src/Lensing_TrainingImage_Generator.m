function [IMS,PARAMS,src_pars,log_kappa] = Lensing_TrainingImage_Generator(nsample,random_seed, WRITE , test_or_train)

%WRITE = 1
%test_or_train = 'test'
if ~exist('WRITE','var')
    WRITE = 0;
end


rng(random_seed)


zLENS = 0.5;
zSOURCE = 2.00;

h=0.71;
OmegaC=0.222;
OmegaLambda=0.734;
sigNORM=0.801;
OmegaBaryon=0.0449;
OmegaM=OmegaC+OmegaBaryon;
c = 2.998E8; G = 6.67259E-11;  Msun= 1.98892e30;
pc = 3.0857E16; kpc=1e3*pc; Mpc=1e6*pc;
H0=100*h*(1000/Mpc); %in units of 1/s
rhocrit=(3*H0^2)/(8*pi*G);   %SI:  kg/m^3
H=100*h;
Ds = (1e-6) * AngularDiameter(0,zSOURCE);    %in Mpc
Dd = (1e-6) * AngularDiameter(0,zLENS);    %in Mpc
Dds = (1e-6) * AngularDiameter(zLENS,zSOURCE);    %in Mpc




extend_ratio = 1.6;

num_im_pix = 400;
num_im_output_pix = 192;
imside = extend_ratio * num_im_output_pix * 0.04;
imside_out = num_im_output_pix * 0.04;

[XIM , YIM]=meshgrid(linspace(-imside/2,imside/2,num_im_pix).*pi./180./3600,linspace(-imside/2,imside/2,num_im_pix).*pi./180./3600);

[X_out , Y_out]=meshgrid(linspace(-imside_out/2,imside_out/2,num_im_output_pix).*pi./180./3600,linspace(-imside_out/2,imside_out/2,num_im_output_pix).*pi./180./3600);




R_ein = zeros(nsample,1);
elp = zeros(nsample,1);
angle = zeros(nsample,1);
elp_x = zeros(nsample,1);
elp_y = zeros(nsample,1);
shear = zeros(nsample,1);
shear_angle = zeros(nsample,1);
xsource = zeros(nsample,1);
ysource = zeros(nsample,1);
XLENS = zeros(nsample,1);
YLENS = zeros(nsample,1);
src_pars = zeros(nsample,4);
magnification = zeros(nsample,1);

% datapath = getenv('Ensai_lens_training_dataset_path');
datapath = [getenv('LOCAL_SCRATCH') '/SAURON/ARCS_2/'];
galaxy_image_path = [getenv('LOCAL_SCRATCH') '/Small_Galaxy_Zoo/'];
mkdir(datapath)
file_list = ls(galaxy_image_path);
file_names = strsplit(file_list);
file_names = sortrows(file_names.',1);
file_names(1)=[];
n_GalZoo_sample = numel(file_names);
file_inds = 1:numel(file_names);
src_numpix = 212;
n_source_sample = n_GalZoo_sample;

%load([getenv('SCRATCH') '/GREAT_gal_ims.mat'],'gal_ims');
%n_source_sample = numel(gal_ims);

disp('loading...')
load([getenv('LOCAL_SCRATCH') '/GREAT_IMS30.mat'],'GREAT_IMS');
n_source_sample = size(GREAT_IMS,1)
disp('done')

[xsrc , ysrc] = meshgrid(linspace(-1,1,src_numpix).*pi./180./3600);
rfilt = sqrt(xsrc.^2+ysrc.^2);
taper = 1./(1+(rfilt./(0.6.*pi./180./3600)).^6); taper = taper./max(taper(:));


if random_seed==-1
   rng('shuffle')
else
   rng(random_seed)
end

max_R_ein = 3.0 * (num_im_output_pix/192);
max_elp = 0.9;



IMS = zeros(num_im_output_pix,num_im_output_pix);
log_kappa = zeros(num_im_output_pix,num_im_output_pix);



for i = 1:nsample

    if mod(i,200)==0
        disp(i./nsample)
    end
    
    newflux = 0;
    oldflux = 1.0;
    magnification(i) = 0;
    SKY_IM = 0;
    while newflux<(0.99 * oldflux) || (magnification(i)<2.0) || max(SKY_IM(:))==0



    
    R_ein(i) = 0.1 + rand(1) .* (max_R_ein-0.1);
    elp(i) = rand(1) * max_elp ;
    angle(i) = rand(1).*360;
    XLENS(i) = (rand(1)-0.5) .* 0.1;
    YLENS(i) = (rand(1)-0.5) .* 0.1;
    shear(i) =  rand(1) .* 0.0 ;
    shear_angle(i) =  rand(1) .* 0.0;
    
    
    image_ind = ceil(rand(1).*n_source_sample);
    size_scale = rand(1).*0.8+0.2;
    
    src_rad = 0.5.*size_scale;
    
    [XS ,YS , ~ ,sigma_n ,MSIS] = SIE_RayTrace_fromRein(XIM,YIM,XIM,YIM,R_ein(i),elp(i),shear(i),shear_angle(i),zLENS,zSOURCE,angle(i),[XLENS(i) YLENS(i)]);
    
    %     kappa_map = get_SIE_kappa( XIM , YIM , sigma_n , elp(i) , angle(i) , XLENS(i) , YLENS(i) , zLENS , zSOURCE );
    %     kappa_map = imresize(kappa_map,[num_im_output_pix num_im_output_pix],'box');
    %     log_kappa(:,:,i) = log10(kappa_map);
    
    
    
    
    [~,~,X1,Y1,X2,Y2] = Caustic_Analytical(MSIS,elp(i),'arcsec','r',0.5,2.0,angle(i),1,[XLENS(i) YLENS(i)]);
    
   
    SKY_IM = 0;
   
    xsource(i) = 10;
    ysource(i) = 10;
    if rand(1)<0.5
        
        xCen = mean(X2(:));
        yCen = mean(Y2(:));
        
        [THET,RHO]=cart2pol(X2-xCen,Y2-yCen);
        [X2_UP,Y2_UP]=pol2cart(THET,RHO+0.15);
        X2_UP = X2_UP + xCen;
        Y2_UP = Y2_UP + yCen;
        max_source_xy_range = max(max(X2_UP(:))-min(X2_UP(:)) , max(Y2_UP(:))-min(Y2_UP(:))) .* 1;
        
        
        while ~inpolygon(xsource(i),ysource(i),X2_UP,Y2_UP)
            xsource(i) = (rand(1)-0.5).* max_source_xy_range + xCen;
            ysource(i) = (rand(1)-0.5).* max_source_xy_range + yCen;
        end
        
        
    else
        xCen = mean(X1(:));
        yCen = mean(Y1(:));
        
        
        X1 = (X1-xCen).*0.7 + xCen;
        Y1 = (Y1-yCen).*0.7 + yCen;
        
        
        max_source_xy_range = max(max(X1(:))-min(X1(:)) , max(Y1(:))-min(Y1(:))) .* 1;
        while ~inpolygon(xsource(i),ysource(i),X1,Y1)
            xsource(i) = random('norm',xCen,max_source_xy_range./5);
            ysource(i) = random('norm',yCen,max_source_xy_range./5);
            
        end
        
        
    end
    
    
    src_pars(i,:) = [image_ind xsource(i) ysource(i) size_scale];
    
    

    %scurr = rng;
    %Nclump=ceil(rand(1).*5);
    %rndmat = rand(2);
    %cov=rndmat.'*rndmat;
    %cov = cov./((cov(1,1)+cov(2,2))./2) .* (src_rad./4)^2;
    %COORDS = mvnrnd(zeros(Nclump,2),cov);
    %temp_src_im = clumpy_source(COORDS(:,1),COORDS(:,2),0.01+rand(1,Nclump).*0.1,src_numpix,2.0,rand(1,Nclump));
    %rng(scurr)

    temp_src_im = GREAT_IMS{image_ind};
    imindx = ceil( rand(1) * size(temp_src_im,3) );
    temp_src_im = temp_src_im(:,:,imindx);

    %temp_src_im = gal_ims{image_ind};
    %temp_src_im = double(imread([galaxy_image_path file_names{file_inds(image_ind)}]))./255;
    temp_src_im = imresize(temp_src_im,[src_numpix src_numpix]);
    temp_src_im = temp_src_im./max(temp_src_im(:));
    source_image = temp_src_im .* taper;
    
    
    

        SKY_IM_L = interp2(xsrc.*size_scale+(xsource(i)).*pi./180./3600,ysrc.*size_scale+(ysource(i)).*pi./180./3600,source_image,XS,YS,'linear',0);
        unlensed_flux = sum(source_image(:))./numel(source_image)./ (imside/(2*size_scale))^2 .* numel(SKY_IM_L);        
        magnification(i) = sum(SKY_IM_L(:)) ./ unlensed_flux;
        
        SKY_IM = interp2(XIM,YIM,SKY_IM_L, X_out , Y_out , 'linear', 0);
        
        newflux = sum(SKY_IM(:))./extend_ratio^2 .* (num_im_pix/num_im_output_pix)^2;
        oldflux = sum(SKY_IM_L(:));
        
    end 
   
    if max(SKY_IM(:))~=0
        SKY_IM = SKY_IM./max(SKY_IM(:));
    end

    IMS = SKY_IM;
    
    %     imshow(SKY_IM,[],'xdata',[-3 3],'ydata',[-3 3]);
    %     pause;
    if WRITE==1
        imwrite(SKY_IM,[datapath  test_or_train '_' num2str(i,'%.7d') '.png'],'bitdepth',16);
    end
    
    
end


Q = [R_ein elp angle XLENS YLENS shear shear_angle];
PARAMS = [map_parameters(Q,'code')  magnification];

if WRITE==1
    dlmwrite([datapath 'parameters_' test_or_train '.txt'],PARAMS,' ')
    dlmwrite([datapath 'parameters_source_' test_or_train '.txt'],src_pars,' ')
end



end






function p = map_parameters(q,code_or_decode)

SCALE_SHEAR = 5;

if strcmp(code_or_decode,'code')

    Rein = q(:,1);
    elp_x = q(:,2).* cos(q(:,3).*pi./180);
    elp_y = q(:,2).* sin(q(:,3).*pi./180);
    xlens = q(:,4);
    ylens = q(:,5);
    shear_x = SCALE_SHEAR .* q(:,6).* cos(q(:,7).*pi./180);
    shear_y = SCALE_SHEAR .* q(:,6).* sin(q(:,7).*pi./180);
    p = [Rein   elp_x    elp_y    xlens    ylens shear_x shear_y];

elseif strcmp(code_or_decode,'decode')
    
    Rein = q(:,1);
    elp = sqrt(q(:,2).^2+q(:,3).^2) ;
    angle = atan2(q(:,3),q(:,2)) .* 180./pi;
    xlens = q(:,4);
    ylens = q(:,5);
    shear = sqrt(q(:,6).^2+q(:,7).^2) ./ SCALE_SHEAR;
    shear_angle = atan2(q(:,7),q(:,6)).* 180./pi;
    p = [Rein   elp    angle    xlens    ylens shear shear_angle];

end

end







function [xsource ysource R_ein sigma_cent Minterior]=SIE_RayTrace_fromRein(ximage,yimage,xsource0,ysource0,REIN,elpSIS,Ext_Shear,Shear_angle,zLENS,zSOURCE,theta,offsets)

h=0.71;
OmegaC=0.222;
OmegaLambda=0.734;
sigNORM=0.801;
OmegaBaryon=0.0449;
OmegaM=OmegaC+OmegaBaryon;
c = 2.998E8; G = 6.67259E-11;  Msun= 1.98892e30;
pc = 3.0857E16; kpc=1e3*pc; Mpc=1e6*pc;
H0=100*h*(1000/Mpc); %in units of 1/s
rhocrit=(3*H0^2)/(8*pi*G);   %SI:  kg/m^3
H=100*h;

Ds = (1e-6) * AngularDiameter(0,zSOURCE);    %in Mpc
Dd = (1e-6) * AngularDiameter(0,zLENS);    %in Mpc
Dds = (1e-6) * AngularDiameter(zLENS,zSOURCE);    %in Mpc

sigma_cent = sqrt(299800000^2/(4*pi).* REIN .*pi./180./3600 .* AngularDiameter(0,zSOURCE)/AngularDiameter(zLENS,zSOURCE));
MSIS = (pi*(sigma_cent^2)*(REIN.*pi./180./3600)*Dd*Mpc)/G/Msun;

if isempty(xsource0) || isempty(ysource0)
    xsource0=ximage;
    ysource0=yimage;
end
if Ext_Shear==0
    Ext_Shear=1e-15;
end

% xoffset=-offsets(1); yoffset=-offsets(2);
xoffset=-offsets(1)*(pi/3600/180); yoffset=-offsets(2)*(pi/3600/180);
COX=xsource0-ximage;
COY=ysource0-yimage;

% A=0; B=0; C=0; D=0;
%Ray trace an SIE based on analytical deflection. field: in arcsec

theta=theta*(pi/180);
Shear_angle=Shear_angle*(pi/180);





xsource=zeros(size(ximage));
ysource=zeros(size(yimage));
% Potential=zeros(size(yimage));

f=1-elpSIS;
fp=sqrt(1-f^2);
% sigma_cent=sigma(MSIS,zLENS);
sigma_cent = ((MSIS*G*Msun*Ds)/(4*pi^2*Dd*Dds*Mpc))^(1/4)*sqrt(c);

% sigma_cent = MSIS;
R_ein = 4 * pi * (sigma_cent/c)^2 *(Dds/Ds);
% Minterior=(pi*(sigma_cent^2)*R_ein*Dd*Mpc)/G/Msun;
ZXI_0 = 4 * pi * (sigma_cent/c)^2 *(Dd*Dds/Ds);
% R_ein=4 * pi * (sigma_cent/c)^2 *(Dds/Ds);
% R_ein*(60*60*180/pi)
% return

ximage=ximage+xoffset;
yimage=yimage+yoffset;



if theta~=0
    [TH RHO]=cart2pol(ximage,yimage);
    [ximage,yimage] = pol2cart(TH-theta,RHO);
end



Shear_angle=Shear_angle-theta;
g1=Ext_Shear*(-cos(2*Shear_angle));
g2=Ext_Shear*(-sin(2*Shear_angle));
g3=Ext_Shear*(-sin(2*Shear_angle));
g4=Ext_Shear*( cos(2*Shear_angle));

par=atan2(yimage,ximage);
xsource=((Dd.*ximage./ZXI_0-(sqrt(f)/fp).*asinh(cos(par).*fp./f)).*ZXI_0./Dd)-((g1.*ximage)+(g2.*yimage));
ysource=((Dd.*yimage./ZXI_0-(sqrt(f)/fp).*asin(fp.*sin(par))).*ZXI_0./Dd)-((g3.*ximage)+(g4.*yimage));


% subplot(2,2,1)
% imshow(abs((sqrt(f)/fp).*asinh(cos(par).*fp./f)).*ZXI_0./Dd,[]);colormap(jet);colorbar
% subplot(2,2,2)
% imshow(abs((sqrt(f)/fp).*asin(fp.*sin(par))).*ZXI_0./Dd,[]);colormap(jet);colorbar

% parfor i=1:numel(ximage)
%         par=atan2(yimage(i),ximage(i));
%         xsource(i)=((Dd.*ximage(i)./ZXI_0-(sqrt(f)/fp)*asinh(cos(par)*fp/f))*ZXI_0/Dd)-((g1.*ximage(i))+(g2.*yimage(i)));
%         ysource(i)=((Dd.*yimage(i)./ZXI_0-(sqrt(f)/fp)*asin(fp*sin(par)))*ZXI_0/Dd)-((g3.*ximage(i))+(g4.*yimage(i)));

%         xsource(i)=((Dd.*ximage(i)./ZXI_0-(sqrt(f)/fp)*asinh(cos(par)*fp/f))*ZXI_0/Dd);
%         ysource(i)=((Dd.*yimage(i)./ZXI_0-(sqrt(f)/fp)*asin(fp*sin(par)))*ZXI_0/Dd);

%         xalpha(i)=((sqrt(f)/fp)*asinh(cos(par)*fp/f))*ZXI_0/Dd;
%         yalpha(i)=((sqrt(f)/fp)*asin(fp*sin(par)))*ZXI_0/Dd;


%         DeltA=sqrt(cos(par)^2+(f^2*sin(par)^2));
%         x1=(Dd.*ximage(i)./ZXI_0); x2=(Dd.*yimage(i)./ZXI_0);
%         x=sqrt(x1^2+(f^2*x2^2))/DeltA;

%         Potential(i)=(sqrt(f)/fp)*x*((abs(sin(par))*acos(DeltA))+(abs(cos(par))*acosh(DeltA/f)));

% end;



if theta~=0
    
    [TH RHO]=cart2pol(ximage,yimage);
    [ximage,yimage] = pol2cart(TH+theta,RHO);
    
    [TH RHO]=cart2pol(xsource,ysource);
    [xsource,ysource] = pol2cart(TH+theta,RHO);
end



ximage=ximage-xoffset;
yimage=yimage-yoffset;
xsource=xsource-xoffset+COX;
ysource=ysource-yoffset+COY;




% if strcmp(action,'plot')
%     [xsource ysource]=TrianGulate(xsource,ysource);
%     [ximage yimage]=TrianGulate(ximage,yimage);
%     hold on
%     plot(ximage,yimage,'color',[0.3 0.3 0.3]);
%     plot(xsource,ysource,'b')
%     axis equal
% elseif strcmp(action,'save')
%     if strcmp(Del,'delaunay')
%         [xsource ysource]=TrianGulate(xsource,ysource);
%         [ximage yimage]=TrianGulate(ximage,yimage);
%     end
% %     save([SavePath '/SIE_RayTracedXY.mat'],'ximage','yimage','xsource','ysource','Nmat','MSIS','elpSIS','R_ein');
%     save(SavePath,'ximage','yimage','xsource','ysource','Nmat','MSIS','elpSIS','R_ein','zLENS','zSOURCE','theta');
% elseif strcmp(action,'pass')
%     if strcmp(Del,'delaunay')
%         [xsource ysource]=TrianGulate(xsource,ysource);
%         [ximage yimage]=TrianGulate(ximage,yimage);
%     end
%
% %     A=xsource; B=ysource;
% end


Minterior=(pi*(sigma_cent^2)*R_ein*Dd*Mpc)/G/Msun;

% if nargin==16
%     if strcmp(displ,'display')
%         disp(['Einstein Radius = ' num2str(R_ein*(180/pi)*3600,'%3.2f') ' [arcsec]']);
%         disp(['Mass inside the Einstein Radius = ' num2str(Minterior,'%3.2e') ' [M_sun]']);
%     end
% end


R_ein=R_ein.*3600*180/pi;
end



function I = LumProfile(xsource,ysource,prof_type,SourcePars)

if strcmpi(prof_type,'disk')
    
    FLUX=SourcePars(1);
    B=SourcePars(2).*(pi/180/3600);
    xsr=SourcePars(3).*(pi/180/3600);
    ysr=SourcePars(4).*(pi/180/3600);
    srelp=1e-8;
    r=sqrt(((xsource-xsr).^2./(1-srelp))+((ysource-ysr).^2).*(1-srelp));
    
    
    %         A=FLUX/(2*pi*B^2);
    I= (r<=B);
    
    
elseif strcmpi(prof_type,'sersic')
    
    if numel(SourcePars)==5
        
        xsource=xsource./(pi/180/3600);
        ysource=ysource./(pi/180/3600);
        
        I0=SourcePars(1);
        n=SourcePars(2);
        Reff=SourcePars(3);
        k = 2.*n-1./3+4./(405.*n)+46/(25515.*n^2);
        
        %     srelp=SourcePars(4);
        %     theta=SourcePars(5);
        theta=0;
        srelp=0.000001;
        %     xsr=SourcePars(6).*(pi/180/3600);
        %     ysr=SourcePars(7).*(pi/180/3600);
        xsr=SourcePars(4);
        ysr=SourcePars(5);
        
        theta=theta.*(pi/180);
        [TH RHO]=cart2pol(xsource,ysource);
        [xsource,ysource] = pol2cart(TH-theta,RHO);
        [TH RHO]=cart2pol(xsr,ysr);
        [xsr,ysr] = pol2cart(TH-theta,RHO);
        r=sqrt(((xsource-xsr).^2./(1-srelp))+((ysource-ysr).^2).*(1-srelp));
        r = r./Reff;
        
        I= I0 .* exp(-k.*  (r.^(1./n))  );
        
        
    elseif numel(SourcePars)==7
        
        
        xsource=xsource./(pi/180/3600);
        ysource=ysource./(pi/180/3600);
        
        I0=SourcePars(1);
        n=SourcePars(2);
        Reff=SourcePars(3);
        k = 2.*n-1./3+4./(405.*n)+46/(25515.*n^2);
        
        srelp=SourcePars(4);
        theta=SourcePars(5);
        
        xsr=SourcePars(6);
        ysr=SourcePars(7);
        
        theta=theta.*(pi/180);
        [TH RHO]=cart2pol(xsource,ysource);
        [xsource,ysource] = pol2cart(TH-theta,RHO);
        [TH RHO]=cart2pol(xsr,ysr);
        [xsr,ysr] = pol2cart(TH-theta,RHO);
        r=sqrt(((xsource-xsr).^2./(1-srelp))+((ysource-ysr).^2).*(1-srelp));
        r = r./Reff;
        
        I= I0 .* exp(-k.*  (r.^(1./n))  );
        
        
        
    end
    
    
    
    
    
elseif strcmpi(prof_type,'gaussian')
    
    if numel(SourcePars)==4
        
        FLUX=SourcePars(1);
        B=SourcePars(2).*(pi/180/3600);
        theta=0;
        srelp=1e-9;
        xsr=SourcePars(3).*(pi/180/3600);
        ysr=SourcePars(4).*(pi/180/3600);
        
        r=sqrt(((xsource-xsr).^2./(1-srelp))+((ysource-ysr).^2).*(1-srelp));
        
        
        A=FLUX/(2*pi*B^2);
        I=A.*exp(-0.5.*(r./B).^2);
        
    elseif numel(SourcePars)==6
        
        FLUX=SourcePars(1);
        B=SourcePars(2).*(pi/180/3600);
        srelp=SourcePars(3);
        theta=SourcePars(4);
        xsr=SourcePars(5).*(pi/180/3600);
        ysr=SourcePars(6).*(pi/180/3600);
        
        theta=theta.*(pi/180);
        [TH RHO]=cart2pol(xsource,ysource);
        [xsource,ysource] = pol2cart(TH-theta,RHO);
        [TH RHO]=cart2pol(xsr,ysr);
        [xsr,ysr] = pol2cart(TH-theta,RHO);
        r=sqrt(((xsource-xsr).^2./(1-srelp))+((ysource-ysr).^2).*(1-srelp));
        
        
        A=FLUX/(2*pi*B^2);
        I=A.*exp(-0.5.*(r./B).^2);
        
    end
    
end

end





function out_vec = AngularDiameter(Zd,z)
%Return the angular diameter distance in parsecs

NLENGTH = max(length(Zd),length(z));
out_vec = zeros(NLENGTH,1);
if length(Zd)==1 & length(z)~=1
    Zd = ones(size(z)).*Zd;
elseif length(Zd)~=1 & length(z)==1
    z = ones(size(Zd)).*z;
end

h=0.71;
OmegaC=0.222;
OmegaLambda=0.734;
sigNORM=0.801;
OmegaBaryon=0.0449;
OmegaM=OmegaC+OmegaBaryon;
c = 2.998E8; G = 6.67259E-11;  Msun= 1.98892e30;
pc = 3.0857E16; kpc=1e3*pc; Mpc=1e6*pc;
H0=100*h*(1000/Mpc); %in units of 1/s
rhocrit=(3*H0^2)/(8*pi*G);   %SI:  kg/m^3
H=100*h;

H0 = 100*h;   % Hubble constant
WM = OmegaM;   % Omega(matter)
WV = OmegaLambda;   % Omega(vacuum) or lambda



% initialize constants

c = 299792.458; % velocity of light in km/sec
DTT = 0.5;      % time from z to now in units of 1/H0
DCMR = 0.0;     % comoving radial distance in units of c/H0


for iJ=1:NLENGTH
    
    h = H0/100.;
    WR = 4.165E-5/(h*h);     %   # includes 3 massless neutrino species, T0 = 2.72528
    WK = 1-WM-WR-WV;
    az = 1.0/(1+1.0*z(iJ));
    aZd = 1.0/(1+1.0*Zd(iJ));
    n=2e4;   %     # number of points in integrals
    
    %# do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
    
    for a=az:(1-az)/n:aZd,
        adot = sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a));
        DTT = DTT + 1./adot;
        DCMR = DCMR + 1./(a*adot);
    end;
    
    
    
    
    DCMR = (1.-az)*DCMR/n;
    
    
    
    x = sqrt(abs(WK))*DCMR;
    if x > 0.1
        if WK > 0
            ratio =  0.5*(exp(x)-exp(-x))/x;
        else
            ratio = sin(x)/x;
        end;
    else
        y = x*x;
        if WK < 0
            y = -y;
        end;
        ratio = 1. + y/6. + y*y/120.;
    end;
    
    
    
    DCMT = ratio*DCMR;
    DA = az*DCMT;
    DA_Mpc = (c/H0)*DA;
    DA = DA_Mpc * 1e6;
    out = DA;                     %in Parsecs
    if (Zd(iJ)==z(iJ))
        out=0;
    end;
    
    
    out_vec(iJ) = out;
    
end

end



function kappa = get_SIE_kappa( XIM , YIM , sigma_n , elp , angle , XLENS , YLENS , zLENS , zSOURCE )

h=0.71;
OmegaC=0.222;
OmegaLambda=0.734;
sigNORM=0.801;
OmegaBaryon=0.0449;
OmegaM=OmegaC+OmegaBaryon;
c = 2.998E8; G = 6.67259E-11;  Msun= 1.98892e30;
pc = 3.0857E16; kpc=1e3*pc; Mpc=1e6*pc;
H0=100*h*(1000/Mpc); %in units of 1/s
rhocrit=(3*H0^2)/(8*pi*G);   %SI:  kg/m^3
H=100*h;



Ds = AngularDiameter(0,zSOURCE);
Dds = AngularDiameter(zLENS,zSOURCE);

theta = angle.*pi./180;
[TH , RHO]=cart2pol(XIM,YIM);
[XIM,YIM] = pol2cart(TH-theta,RHO);

[TH , RHO]=cart2pol(XLENS,YLENS);
[XLENS,YLENS] = pol2cart(TH-theta,RHO);

r = sqrt((XIM-XLENS.*pi./180./3600).^2+((YIM-YLENS.*pi./180./3600).*(1-elp)).^2) ;
R_ein = 4.*pi .* (sigma_n/c).^2 * Dds./Ds;
kappa = sqrt(1-elp) .* R_ein./(2.*r);

end


function [alphaX , alphaY]=RayTraceMap(kappa,ximage,yimage)


h = waitbar(0,'Ray tracing...');

dx = yimage(2) - yimage(1);

alphaX=zeros(size(ximage));
alphaY=zeros(size(ximage));
% smoothing_length = 1e-14 .* pi./180./3600;
% inds=find(mask==1).';
n=0;
% for i=inds
for i=1:numel(ximage)
    n=n+1;
    n
    %     i
    if mod(i,200)==0
        waitbar(i./numel(ximage),h,sprintf('%4.0f %% Completed',i./numel(ximage)*100))
    end
    
    xp=ximage-ximage(i);
    yp=yimage-yimage(i);
    %     rp=sqrt((xp+smoothing_length).^2+(yp+smoothing_length).^2);
    rp=sqrt((xp).^2+(yp).^2);
    
    
    AX = -(1./pi) .*   kappa .* (xp)./(rp.^2) .*dx^2 ;
    AY = -(1./pi) .*   kappa .* (yp)./(rp.^2) .*dx^2 ;
    AX(rp==0)=0;
    AY(rp==0)=0;
    alphaX(i) = sum( AX(:) );
    alphaY(i) = sum( AY(:) );
    
    %     alphaX(i) = -(1./pi) .*  sum(sum( kappa .* (xp)./(rp.^2) .*dx^2 ));
    %     alphaY(i) = -(1./pi) .*  sum(sum( kappa .* (yp)./(rp.^2) .*dx^2 ));
    
end


close(h)

xsource = ximage - alphaX;
ysource = yimage - alphaY;




end


function [ h1 , h2 , X1 , Y1 , X2 , Y2 ]=Caustic_Analytical(LensM,elp,units,CoL,zLENS,zSOURCE,theta,scale_me,offsets)
h1=0; h2=0;

X1 = 0;
Y1 = 0;
X2 = 0;
Y2 = 0;


theta=-theta+90;

h=0.71;
OmegaC=0.222;
OmegaLambda=0.734;
sigNORM=0.801;
OmegaBaryon=0.0449;
OmegaM=OmegaC+OmegaBaryon;
c = 2.998E8; G = 6.67259E-11;  Msun= 1.98892e30;
pc = 3.0857E16; kpc=1e3*pc; Mpc=1e6*pc;
H0=100*h*(1000/Mpc); %in units of 1/s
rhocrit=(3*H0^2)/(8*pi*G);   %SI:  kg/m^3
H=100*h;


Dd =AngularDiameter(0,zLENS)*1e-6;
Dds=AngularDiameter(zLENS,zSOURCE)*1e-6;
Ds =AngularDiameter(0,zSOURCE)*1e-6;
% sig=sigma(LensM,zLENS);
sig=((LensM*G*Msun*Ds)/(4*pi^2*Dd*Dds*Mpc))^(1/4)*sqrt(c);
xoff=offsets(1);
yoff=offsets(2);
ZXI_0 = 4 * pi * (sig/c)^2 *(Dd*Dds/Ds);


f=1-elp;
fp=sqrt(1-f^2);
par=linspace(0,2*pi,200);
x=(-sqrt(f)/fp).*asinh(cos(par).*fp./f);
y=(-sqrt(f)/fp).*asin(sin(par).*fp);


delta=sqrt(cos(par).^2+f^2.*sin(par).^2);
x1=((sqrt(f)./delta).*cos(par))-((sqrt(f)/fp).*asinh(cos(par).*fp./f));
y1=((sqrt(f)./delta).*sin(par))-((sqrt(f)/fp).*asin(sin(par).*fp));

% xCrit1=((sqrt(f)./delta).*cos(par))-((sqrt(f)/fp).*asinh(cos(par).*fp./f));
% yCrit1=((sqrt(f)./delta).*sin(par))-((sqrt(f)/fp).*asin(sin(par).*fp));

% cr=(4*pi/(mu^2))+((16*sqrt(6)/15)*((1+f)/fp)*(1/(mu^(5/2))));
% cr=cr*((ZXI_0*Ds/Dd)^2);  %cross section in physical units in the source plane
% disp(['Cross-section on steradians: ' num2str(cr/(Ds^2))]);


% con=180*3600/pi;
% con=1;
if nargin<8
    scale_me=1;
end

x=x*ZXI_0;
y=y*ZXI_0;
x1=x1*ZXI_0;
y1=y1*ZXI_0;



% clf
% hold on

if strcmpi(units,'kpc')
    plot(1e3*y,1e3*x,'m','LineWidth',2);
    plot(1e3*y1,1e3*x1,'m','LineWidth',2);

elseif strcmpi(units,'pc')
    plot(1e6*y,1e6*x,'m','LineWidth',2);
    plot(1e6*y1,1e6*x1,'m','LineWidth',2);

elseif strcmpi(units,'rad')
    %     plot(y/Ds,x/Ds,'--y','LineWidth',2);
    %     plot(y1/Ds,x1/Ds,'--y','LineWidth',2);
    
    %     plot(y/Dd,x/Dd,CoL,'LineWidth',2);
    %     plot(y1/Dd,x1/Dd,CoL,'LineWidth',2);
    
    %     plot(y/Dd,x/Dd,CoL,'LineWidth',1);
    %     plot(y1/Dd,x1/Dd,CoL,'LineWidth',1);
    if nargin>7
        theta=theta*(pi/180);
        [t r]=cart2pol(x,y);
        [x,y] = pol2cart(t+theta,r);
        [t r]=cart2pol(x1,y1);
        [x1,y1] = pol2cart(t+theta,r);
        
        x=scale_me.*x;
        y=scale_me.*y;
        x1=scale_me.*x1;
        y1=scale_me.*y1;
        
        X1 = ((y/Dd)+xoff);
        Y1 = ((x/Dd)+yoff);
        X2 = ((y1/Dd)+xoff);
        Y2 = ((x1/Dd)+yoff);
        
        h1=plot(y/Dd+xoff,x/Dd+yoff,'Color',CoL,'LineWidth',3,'linestyle','-');
        hold on
        h2=plot(y1/Dd+xoff,x1/Dd+yoff,'Color',CoL,'LineWidth',3,'linestyle','-');
    else
        x=scale_me.*x;
        y=scale_me.*y;
        x1=scale_me.*x1;
        y1=scale_me.*y1;
        
        h1=plot(y/Dd,x/Dd,'Color',CoL,'LineWidth',1.5,'linestyle','-');
        hold on
        h2=plot(y1/Dd,x1/Dd,'Color',CoL,'LineWidth',1.5,'linestyle','-');
    end
elseif strcmpi(units,'arcsec')
    theta=theta*(pi/180);
    [t , r]=cart2pol(x,y);
    [x,y] = pol2cart(t+theta,r);
    [t , r]=cart2pol(x1,y1);
    [x1,y1] = pol2cart(t+theta,r);
    scale_me=(3600*180/pi);
    x=scale_me.*x;
    y=scale_me.*y;
    x1=scale_me.*x1;
    y1=scale_me.*y1;

    X1 = ((y/Dd)+xoff);
    Y1 = ((x/Dd)+yoff);
    X2 = ((y1/Dd)+xoff);
    Y2 = ((x1/Dd)+yoff);
%     h1=plot(((y/Dd)+xoff),((x/Dd)+yoff),'Color',CoL,'LineWidth',1,'linestyle','--');
%     h2=plot(((y1/Dd)+xoff),((x1/Dd)+yoff),'Color',CoL,'LineWidth',1,'linestyle','--');
    
    
end

% axis equal

end




% %%
% disp('Simulating data ...')
% rng(1)
% nsample = 9999;
% logM = zeros(nsample,1);
% elp = zeros(nsample,1);
% angle = zeros(nsample,1);
% xsource = zeros(nsample,1);
% ysource = zeros(nsample,1);
% for i=1:nsample
%     i
%     logM(i) = rand(1).*(11.8-10.5)+10.5;
%     elp(i) = rand(1).*0.6;
%     angle(i) = rand(1).*90;
% %     XY = datasample([0.1 0.1; -0.1 0.1 ; 0.1 -0.1; -0.1 -0.1],1);
% %     [XY(1),XY(2)] = pol2cart(rand(1).*2*pi,0.1.*sqrt(2));
%     xsource(i) = (rand(1)-0.5).*0.4;
%     ysource(i) = (rand(1)-0.5).*0.4; %datasample([-0.1 0.1],1); %(rand(1)-0.5).*0.1;
%
%
%
%     [XS ,YS ,R_ein ,sigma_cent ,Minterior]=SIE_SUB_RayTrace(XIM,YIM,XIM,YIM,10^logM(i),elp(i),1e-20,0,zLENS,zSOURCE,angle(i),[0 0]);
%     SKY_IM = LumProfile(XS,YS,'gaussian',[1 0.05 xsource(i) ysource(i)]);
%     SKY_IM = SKY_IM./max(SKY_IM(:));
%
% %     FFT_IM = fftshift(fft2(fftshift(SKY_IM)));
% %     mAx = max([real(FFT_IM(:)); imag(FFT_IM(:))]);
% %     imwrite(real(FFT_IM)./mAx,[datapath 'FFT_real_lens_im96__' num2str(i,'%.4d') '.png']);
% %     imwrite(imag(FFT_IM)./mAx,[datapath 'FFT_imag_lens_im96__' num2str(i,'%.4d') '.png']);
%
% %     imshow(SKY_IM,[],'xdata',[min(XIM(:)) max(XIM(:))].*3600.*180./pi,'ydata',[min(YIM(:)) max(YIM(:))].*3600.*180./pi); colormap(jet)
% %     pause(0.05)
%
%     imwrite(SKY_IM,[datapath 'lens_im96__' num2str(i,'%.4d') '.png']);
% end
% dlmwrite([datapath 'logM96.txt'],(logM-10.5)./1.3,' ')
% dlmwrite([datapath 'elp96.txt'],elp./0.6,' ')
% dlmwrite([datapath 'angle96.txt'],angle.*0.01./0.9,' ')
% % hold on
% % plot(subhalo_pars(3),subhalo_pars(4),' +r','markersize',10)
%
%
% %%
%
% disp('Simulating data ...')
% rng(2)
% nsample = 999;
% logM = zeros(nsample,1);
% elp = zeros(nsample,1);
% angle = zeros(nsample,1);
% xsource = zeros(nsample,1);
% ysource = zeros(nsample,1);
%
% for i=1:nsample
%     i
%     logM(i) = rand(1).*(11.8-10.5)+10.5;
%     elp(i) = rand(1).*0.6;
%     angle(i) = rand(1).*90;
% %     XY = datasample([0.1 0.1; -0.1 0.1 ; 0.1 -0.1; -0.1 -0.1],1);
% %     [XY(1),XY(2)] = pol2cart(rand(1).*2*pi,0.1.*sqrt(2));
%     xsource(i) = (rand(1)-0.5).*0.4;
%     ysource(i) = (rand(1)-0.5).*0.4; %datasample([-0.1 0.1],1); %(rand(1)-0.5).*0.1;
%
%
%
%     [XS ,YS ,R_ein ,sigma_cent ,Minterior]=SIE_SUB_RayTrace(XIM,YIM,XIM,YIM,10^logM(i),elp(i),1e-20,0,zLENS,zSOURCE,angle(i),[0 0]);
%     SKY_IM1 = LumProfile(XS,YS,'gaussian',[1 0.02 xsource(i) ysource(i)]);
%     SKY_IM2 = LumProfile(XS,YS,'gaussian',[1 0.02 xsource(i)+0.06 ysource(i)+0.06]);
%     SKY_IM3 = LumProfile(XS,YS,'gaussian',[1 0.02 xsource(i)+0.06 ysource(i)-0.06]);
%     SKY_IM = SKY_IM1 + SKY_IM2 + SKY_IM3;
%     SKY_IM = SKY_IM./max(SKY_IM(:));
%
% %     FFT_IM = fftshift(fft2(fftshift(SKY_IM)));
% %     mAx = max([real(FFT_IM(:)); imag(FFT_IM(:))]);
% %     imwrite(real(FFT_IM)./mAx,[datapath 'FFT_test_lens_im96__' num2str(i,'%.4d') '.png']);
% %     imwrite(imag(FFT_IM)./mAx,[datapath 'FFT_test_lens_im96__' num2str(i,'%.4d') '.png']);
%
%     imshow(SKY_IM,[],'xdata',[min(XIM(:)) max(XIM(:))].*3600.*180./pi,'ydata',[min(YIM(:)) max(YIM(:))].*3600.*180./pi); colormap(jet)
%     pause;
%
% %     imwrite(SKY_IM,[datapath 'test_lens_im96__' num2str(i,'%.4d') '.png']);
% end
% % dlmwrite([datapath 'test_logM96.txt'],(logM-10.5)./1.3,' ')
% % dlmwrite([datapath 'test_elp96.txt'],elp./0.6,' ')
% % dlmwrite([datapath 'test_angle96.txt'],angle.*0.01./0.9,' ')
% % hold on
% % plot(subhalo_pars(3),subhalo_pars(4),' +r','markersize',10)
%
%
% %%
% figure
% logM = load([datapath 'logM96.txt']);
% elp = load([datapath 'elp96.txt']);
% angle = load([datapath 'angle96.txt']);
% predict=load([datapath 'predict_mea.txt']);
% clf
% s(1) = subplot(3,1,1);
% plot(predict(:,1))
% hold on
% plot(logM,'--')
%
% s(2) = subplot(3,1,2);
% plot(predict(:,2))
% hold on
% plot(elp,'--')
%
% s(3) = subplot(3,1,3);
% plot(predict(:,3))
% hold on
% plot(angle,'--')
%
% linkaxes(s,'x');
%
% %%
%
% figure
% logM=load([datapath 'test_logM96.txt']);
% elp=load([datapath 'test_elp96.txt']);
% angle=load([datapath 'test_angle96.txt']);
% predict=load([datapath 'test_predict_mea.txt']);
% clf
% s(1) = subplot(3,1,1);
% plot(predict(:,1))
% hold on
% plot(logM,'--')
%
% s(2) = subplot(3,1,2);
% plot(predict(:,2))
% hold on
% plot(elp,'--')
%
% s(3) = subplot(3,1,3);
% plot(predict(:,3))
% hold on
% plot(angle,'--')
%
% linkaxes(s,'x');
