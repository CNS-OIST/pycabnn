% Example MATLAB script to generate coordinates of mossy fiber rosettes (GLpoints.dat)
%
% Written by Shyam Kumar Sudhakar
% Computational Neuroscience Unit, Okinawa Institute of Science and Technology, Japan
% Supervisor: Erik De Schutter
%
% Correspondence: Shyam Kumar Sudhakar (shyamk@umich.edu)
%
% September 16, 2017

%rng(252)
s = RandStream('mt19937ar','Seed',252);
RandStream.setDefaultStream(s);

Transverse_range = 1500; %% xrange
Horizontal_range = 700;  %% yrange
Vertical_range= 200%;140 for daria;  %%% zrange
%nMF= 2; %numMF %considerin 1000*600 5000/mm2
z_mid= randi([50,150]); %% parallel line to parasagittal to determine the location of pts along z

semi_x = 64/2;%297/2; % semi x axis of high density ellipse
semi_y = 84/2;%474/2;% semi y axis of high density ellipse
semi_z = 50; %% semi z axis of high density ellipse
insvolc=0; %%num of rosettes inside volume of 1000*600*200
box_fac=2.5; % ratio of cuboid varion of y/ variation of x (exp data) %% sugihara & shinoda
Box_ld_x = 64+40;%297+40; %% x dimension of low density cuboid
Box_ld_y = 84+40*box_fac;%474+40*box_fac; %% y dimension of low density cuboid
Box_ld_z = semi_z*2; %% z dimension of low density cuboid
offset = 0;%Box_ld_y/2; %% offset between ell and cuboid along y
k=0;
MF_coordinates=[];%% coordinates of Mf
MF_GL=[];%% coordinates of rosettes
numRosetteperMF_mu=750;%154; %1020%154;
numRosetteperMF_sd=37;numpmperMF_mu=7;numpmperMF_sd=1;

fileID = fopen('GLpoints.dat','w');


% for i=1:nMF %% generate MF coordinates
%     MF_coordinates(i,1) = randi([0,Transverse_range]);
%     MF_coordinates(i,2) = randi([0,Horizontal_range]);
% end

MF_coordinates=dlmread('MFCr.dat');
nMF=length(MF_coordinates);


for i=1:nMF %%for all MF
    i
    %numRosetteperpm = ceil(normrnd(numRosetteperMF_mu,numRosetteperMF_sd)/normrnd(numpmperMF_mu,numpmperMF_sd));
    numRosetteperpm = ceil(normrnd(numRosetteperMF_mu,37)/numpmperMF_mu);
    nGL=ceil((Box_ld_y/1200)*numRosetteperpm); %% num of rosettes =yvariation in model/yvariation of exp data of 1 primary collateral/num of pts of 1 primary collateral
    nGL_hd = ceil(0.8*nGL); %% 80% high density pts
    nGL_ld = nGL-nGL_hd; %% 20% low density pts
    z_mid= randi([50,150]); %% parallel line to parasagittal to determine the location of pts along z

    Ell_xo = MF_coordinates(i,1); %centre x of high density ellipse
    Ell_yo = MF_coordinates(i,2)+offset; %centre y of high density ellipse offset is zero erik asked 2 remove
    Ell_zo = z_mid; %centre z of high dens ellipse
    GLcount=0; %rosette counter
    hdcount=0;ldcount=0; %high density low density pt counter
    GLcount1=1;

%    [x, y, z] =     ellipsoid(Ell_xo,Ell_yo,Ell_zo,semi_x,semi_y,semi_z,20);%draw ellipsoid
%     SObject = surfl(x,y,z);
%     set(SObject,'facecolor','green','edgecolor','none');
%     alpha(0.4)
%     xlabel('x');ylabel('y');,zlabel('z');
%     figure(1);hold on; plot(MF_coordinates(i,1),MF_coordinates(i,2),'g*');%axis([0 Transverse_range 0 Horizontal_range 0 Vertical_range])
%
%     p1 = [Ell_xo-(Box_ld_x/2),Ell_yo-(Box_ld_y/2), Ell_zo-semi_z];  %8 vertices of cuboid
%     p2 = [Ell_xo+(Box_ld_x/2),Ell_yo-(Box_ld_y/2), Ell_zo-semi_z];
%     p3 = [Ell_xo+(Box_ld_x/2),Ell_yo+(Box_ld_y/2), Ell_zo-semi_z];
%     p4 = [Ell_xo-(Box_ld_x/2),Ell_yo+(Box_ld_y/2), Ell_zo-semi_z];
%     p5 = [Ell_xo-(Box_ld_x/2),Ell_yo-(Box_ld_y/2), Ell_zo+semi_z];
%     p6 = [Ell_xo+(Box_ld_x/2),Ell_yo-(Box_ld_y/2), Ell_zo+semi_z];
%     p7 = [Ell_xo+(Box_ld_x/2),Ell_yo+(Box_ld_y/2), Ell_zo+semi_z];
%     p8 = [Ell_xo-(Box_ld_x/2),Ell_yo+(Box_ld_y/2), Ell_zo+semi_z];
%
%      poly_rectangle(p1, p2, p3, p4); % draw cuboid
%      poly_rectangle(p5, p6, p7, p8);
%      poly_rectangle(p1, p4, p8, p5);
%      poly_rectangle(p2, p3, p7, p6);
%      poly_rectangle(p1, p2, p6, p5);
%      poly_rectangle(p4, p3, p7, p8);
%
%
%      %draw farthest point from the origin point
%     figure(1);hold on; plot3(Ell_xo+ceil((Box_ld_x/2)),MF_coordinates(i,2)+Box_ld_y/2,Ell_zo+semi_z,'p','MarkerSize',15,'MarkerFaceColor','y')
%     figure(1);hold on; plot3(Ell_xo-ceil((Box_ld_x/2)),MF_coordinates(i,2)+Box_ld_y/2,Ell_zo+semi_z,'p','MarkerSize',15,'MarkerFaceColor','r')
% %
%      dis = sqrt((Ell_xo-ceil((Box_ld_x/2))-MF_coordinates(i,1))^2+(Ell_yo+ceil((Box_ld_y/2))-MF_coordinates(i,2))^2+(0-Ell_zo+semi_z)^2)%distan bw origin n farthest pt max cond time
% %


    while GLcount<nGL  %%cont upto GLcount==nGL
    k=k+1;


    GLx = randi([Ell_xo-ceil((Box_ld_x/2)),Ell_xo+ceil((Box_ld_x/2))]);   %%random pt inside cuboid
    %GLy = randi([MF_coordinates(i,2),MF_coordinates(i,2)+Box_ld_y]);%% random pt y inside cuboid
    GLy = randi([Ell_yo-ceil((Box_ld_y/2)),Ell_yo+ceil((Box_ld_y/2))]);   %%random pt inside cuboid

    GLz = randi([Ell_zo-semi_z,Ell_zo+semi_z]); %random z inside cuboid
    find=ifinsideEllipse(Ell_xo,Ell_yo,Ell_zo,GLx,GLy,GLz,semi_x,semi_y,semi_z);%% check if inside ellipse


    if find>0 %if inside ellipse
        if hdcount<nGL_hd
        GLcount=GLcount+1;
        hdcount=hdcount+1;  %update counters
        %figure(1);hold on; plot3(GLx,GLy,GLz,'r+');

        if GLx<=Transverse_range && GLx>0 && GLy<=Horizontal_range && GLy>0 && GLz<=Vertical_range %if inside volume record the coord
        insvolc = insvolc+1;

        MF_GL(i,GLcount1,1)=GLx;
        MF_GL(i,GLcount1,2)=GLy;
        MF_GL(i,GLcount1,3)=GLz;
        GLcount1=GLcount1+1;
        fprintf(fileID,'%d %d %d %d\n',i,MF_GL(i,GLcount1-1,1),MF_GL(i,GLcount1-1,2),MF_GL(i,GLcount1-1,3));
        %figure(1);hold on; plot3(GLx,GLy,GLz,'r+');



        end

        %figure(2);plot(MF_GL(i,GLcount,1),MF_GL(i,GLcount,2),'.','Color','r')
        end

    else
           if  ldcount<nGL_ld %if outside ellipse inside cuboid low density pt
           GLcount=GLcount+1;
           ldcount=ldcount+1;
           %figure(1);hold on; plot3(GLx,GLy,GLz,'*','MarkerSize',15,'MarkerEdgeColor','y');

           if GLx<=Transverse_range && GLx>0 && GLy<=Horizontal_range && GLy>0 && GLz<=Vertical_range %%if inside volume record coord
           insvolc = insvolc+1;
           MF_GL(i,GLcount1,1)=GLx;
           MF_GL(i,GLcount1,2)=GLy;
           MF_GL(i,GLcount1,3)=GLz;
           GLcount1=GLcount1+1;
           fprintf(fileID,'%d %d %d %d\n',i,MF_GL(i,GLcount1-1,1),MF_GL(i,GLcount1-1,2),MF_GL(i,GLcount1-1,3));
           %figure(1);hold on; plot3(GLx,GLy,GLz,'k+');


           end
           MF_GL_ld(ldcount,1)=GLx;
           MF_GL_ld(ldcount,2)=GLy;
           MF_GL_ld(ldcount,3)=GLz;


           %figure(2);hold on;plot(MF_GL(i,GLcount,1),MF_GL(i,GLcount,2),'.','Color','k')

           end
    end
    end
end
fclose(fileID)
clear size
d=dlmread('GLpoints.dat');
size(d)
length(unique(d(:,1)))