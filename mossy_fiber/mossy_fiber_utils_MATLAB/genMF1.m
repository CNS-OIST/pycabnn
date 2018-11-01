% Example MATLAB script to generate coordinates of mossy fibers (MFCr.dat) and
% their spike-time data (datasp.dat, l.dat, activeMfibres1.dat)
%
% Written by Shyam Kumar Sudhakar
% Computational Neuroscience Unit, Okinawa Institute of Science and Technology, Japan
% Supervisor: Erik De Schutter
%
% Correspondence: Shyam Kumar Sudhakar (shyamk@umich.edu)
%
% September 16, 2017

%rng(1)
s = RandStream('mt19937ar','Seed',1);
RandStream.setDefaultStream(s);
clear all
Longaxis  =  1500%185%eval (readParameters ('GoCxrange', 'Parameters.hoc'))  % um
Shortaxis =  700;%185%eval (readParameters ('GoCyrange', 'Parameters.hoc'))  % um

MFdensity = 1650;%5000 ;%cells/mm2%190

close all
clear MF_coordinates
fid = fopen('datasp.dat', 'w');
box_fac=2.5;
Xinstantiate = 64+40;%297+40;
Yinstantiate = 84+40*box_fac;%474+40*box_fac;
numMF=(Longaxis+(2*Xinstantiate))*(Shortaxis+(2*Yinstantiate))*MFdensity*1e-6 %number of MF%150
%numMF=(Longaxis)*(Shortaxis)*MFdensity*1e-6 %number of MF

plotMF=1;
fcoor = fopen('MFCr.dat', 'w');
dt = 0.025;%eval (readParameters ('dt', 'Parameters.hoc') );


%bandwidth =  eval (readParameters ('bandwidth', 'Parameters.hoc'))  % um
%nband     =  eval (readParameters ('nband', 'Parameters.hoc'))
%Scale_factor =  eval (readParameters ('Scale_factor', 'Parameters.hoc'))




%%%%%%%%%generate Mf coordinates


for i=1:numMF
    MF_coordinates(i,1) = randi([0-Xinstantiate,Longaxis+Xinstantiate]);
    MF_coordinates(i,2) = randi([0-Yinstantiate,Shortaxis+Yinstantiate]);
    fprintf(fcoor,'%d %d\n',MF_coordinates(i,1),MF_coordinates(i,2));
end
fclose(fcoor);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
centrex = 750;%Longaxis/2; %um %center of nw
centrey = 150;%Shortaxis/2; %um %centre of nw
radius  = 100; %um % radius of circular kernel 100 control

finalMF=[]; %vector that says whether the MF falls within the circle
finalMF = (MF_coordinates(:,1)-centrex).^2 +(MF_coordinates(:,2)-centrey).^2;
finalMF(finalMF<=radius*radius) = 1;%activated
finalMF(finalMF>radius*radius)  = 0;%inactivated

%%%second spatial kernal 1250,350
finalMF1=[];
centrex1 = 750; %um %center of nw
centrey1 = 350;%Shortaxis/2; %um %centre of nw
finalMF1 = (MF_coordinates(:,1)-centrex1).^2 +(MF_coordinates(:,2)-centrey1).^2;
finalMF1(finalMF1<=radius*radius) = 1;%activated
finalMF1(finalMF1>radius*radius)  = 0;%inactivated
find_ac = find(finalMF1);
finalMF(find_ac)=1;

finalMF2=[];
centrex2 = 750; %um %center of nw
centrey2 = 550;%Shortaxis/2; %um %centre of nw
finalMF2 = (MF_coordinates(:,1)-centrex2).^2 +(MF_coordinates(:,2)-centrey2).^2;
finalMF2(finalMF2<=radius*radius) = 1;%activated
finalMF2(finalMF2>radius*radius)  = 0;%inactivated
find_ac = find(finalMF2);
finalMF(find_ac)=1;



%%%%%%%%%% plot activated and inactivated MF%%%%%%%%%%%%
if plotMF==1
for i=1:numMF
if finalMF(i)==1
hold on,figure(1);plot(MF_coordinates(i,1),MF_coordinates(i,2),'--mo');
xlabel('long axis um');
ylabel('short axis um');
else
hold on,plot(MF_coordinates(i,1),MF_coordinates(i,2),'--ko');
end
end
end
%%%%%%%%%parameters for frequency %%%%%%%%%%%%%%%%%%%%%%%

time_step=dt;
bg_dur=500; %%background firing for activated MF
up_dur=300; %%up phase
down_dur=300; %down pahse
num_epochs = 5;
up_fr= 70;%70;%70;%150;%150;%up phase freq
down_fr=20;%20;%20;%50; %down phase freq
bg_fr=5;%5;%5;%10; %back ground firing phase freq for all MF
bg_start = 0;
totalsimtime = bg_dur+num_epochs*2*300;


up_rate = linspace(up_fr,up_fr,up_dur/0.025); %firing rate vector for up state
down_rate = linspace(down_fr,down_fr,down_dur/0.025); %%firing rate vector for down state
bg_rate_act = linspace(bg_fr,bg_fr,bg_dur/0.025);%background rate for activated MF
bg_rate_inact = linspace(bg_fr,bg_fr,totalsimtime/0.025);%background rate for inactivated MF
epoch_rate = [up_rate down_rate]; %rate vector for one up and down state
epoch_rate = repmat(epoch_rate,[1,num_epochs]); %for n epochs
activated_MF = [bg_rate_act epoch_rate]; %firing rate vector for activated MF
inactivated_MF = bg_rate_inact; % %firing rate vector for inactivated MF


sigma = 2000; %SD of kernal in time domain

size = 10000; %size of the kernel
x = linspace(-size / 2, size / 2, size);
gaussFilter = exp(-x .^ 2 / (2 * sigma ^ 2));
gaussFilter = gaussFilter / sum (gaussFilter); % normalize

activated_sm_MF = conv (activated_MF, gaussFilter, 'same');%smooth activated MF;activated_MF ;
inactivated_sm_MF = inactivated_MF;%conv (inactivated_MF, gaussFilter, 'same');%smooth inactivated MF
figure(2);hold on;plot([0.025:0.025:totalsimtime],activated_sm_MF,'m')
hold on;plot([0.025:0.025:totalsimtime],activated_MF,'r')
ylabel('Frequency Hz');
xlabel('time ms');


for j=1:length(finalMF)
    j
spikeTimes=[];
if finalMF(j)==1 %for activated MF

% for i=1:length(activated_sm_MF)
%                 if activated_sm_MF(i) * time_step / 1000 >= rand(1) % poisson spike if rate*dt>=rand
%                   spikeTimes(end + 1) = i*time_step;
%                 end
% end

        a=0;b=1;
        r = (a + (b-a).*rand(length(activated_sm_MF),1))';
        ind = find((activated_sm_MF.*time_step/1000)>=r);
        spikeTimes = ind*time_step;

		fprintf(fid,'%4.8f\t',spikeTimes);fprintf(fid, '\n'); %write in a file

else
%     for i=1:length(inactivated_sm_MF)
%
%                   if inactivated_sm_MF(i) * time_step / 1000 >= rand(1) %poisson spike if rate*dt>rand
%                   spikeTimes(end + 1) = i*time_step;
%                   end
%     end



        a=0;b=1;
        r = (a + (b-a).*rand(length(inactivated_sm_MF),1))';
        ind = find((inactivated_sm_MF.*time_step/1000)>=r);
        spikeTimes = ind*time_step;

        fprintf(fid,'%4.8f\t',spikeTimes);fprintf(fid, '\n'); %write in a file

end
end

fclose(fid);


%%%%data to run the model
%%doesnt work if thr s silent mf
data=dlmread('datasp.dat');
activeMF=(find(data(:,1)));
fl = fopen('l.dat', 'w');
fa = fopen('activeMfibres1.dat','w');

activeMF(activeMF>0)=1; %0 if MF doesnt fire any spike 1 if it fires atleast a spike

MFcounter=0;
activeMFcount = [];
for m=1:length(activeMF)
    if activeMF(m)>0
        MFcounter = MFcounter+1;
        activeMFcount(m) = MFcounter;
    else
        activeMFcount(m) = 0;
    end
end

for i=1:numMF
lengthMF(i) = length(find(data(i,:))); %length of each and every MF spike train
end

for i=1:numMF
        		fprintf(fl,'%d\t',lengthMF(i));fprintf(fid, '\n'); %write in a file
                fprintf(fa,'%d\t',activeMFcount(i));fprintf(fid, '\n'); %write in a file

end
fclose(fl);
fclose(fa);


%%%%%plot freq%%%%%
MF=1;
clear size
time=totalsimtime;
if MF==1
isi=[];
tisi=[];

MFspiketime = dlmread('datasp.dat');

ISI=[];
MFspikecell=[];
count=1;
[a,b]=size(MFspiketime);
Ncells = 100;



%%%%%%%%%%%%%%%%%%%%%%%population spike%%%%%%%%%%%%%%%%%%%%%%%%
TW=ones(5,1)/5


firings3=[];
fr=[];
for i=1:a
    i
    st= size(tisi);
    isi = diff(nonzeros(MFspiketime(a,2:end)));
    tisi(st(2)+1:st(2)+size(isi))=isi;
     fr(i) = (length(find(MFspiketime(i,:)))-1)/time*1000;
    index=find(MFspiketime(i,2:end)>0)+1;
    firings3=[firings3;[MFspiketime(i,index)',i*ones(length(index),1)]];
end

FRMF=zeros(1,time);
for t=1:time
    t
FRMF(t)=length(find(firings3(:,1)>(t-1) & firings3(:,1)<=t));
end


FRMFB=conv(FRMF,TW,'same');


%figure(591),hold on; subplot(1,2,1);plot(firings3(1:30000,1),firings3(1:30000,2),'.','MarkerSize',2);axis([0 1000 0 5000]);xlabel('time');ylabel('MF ID')

figure(13);hist(fr,40);xlabel('freq Hz');ylabel('numMF');
%subplot(2,2,3);plot(1:time,FRMF,'k');xlabel('time');ylabel('num of spikes');

end
