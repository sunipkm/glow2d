;+
; NAME:
;	GOLDLOS
;
; LICENSE:
;	This software is part of the GLOW model.  Use is governed by the Open Source
; 	Academic Research License Agreement contained in the file glowlicense.txt.
; 	For more information see the file glow.txt.
;
; PURPOSE:
;	Performs spectral synthesis and Line-Of-Sight integrations from GLOW output files
; 	for simulating disk and limb observations by the GOLD spectrograph.
;
; CATEGORY:
;	Simulation
;
; CALLING SEQUENCE:  
;	Main program
;
; INPUTS:
;	GLOW netCDF output file containing global airglow and atmosphere fields for one time frame
;	O2 absorption cross section netCDF file 'o2absxsectfuv.nc'
;	Specified in first lines of program:
;		amodel		Atmosphere model identifier		string
;		gmodel		Airglow model identifier		string
;		directory	Directory for I/O			string
;		filecode	Filename code for I/O			string
;		nstart		number of first file to process		integer
;		nstop		number of last file to process		integer
;		nplots		number of image plots to make		integer
;		calclbhspec	switch for calculating LBH spectrum	integer (0,1)
;		writecube	switch for writing output file		integer (0,1)
;		makeplots	switch for making plots			integer (0,1)
;		writeplots	switch for writing plots to files	integer (0,1)
;		satlat		latitude of satellite (degrees)		float
;		satlon		longitude of satellite (degrees)	float
;		satalt		altitude of satellite (km)		float
;               nwave           number of wavelength bins               integer
;		wavemin		minimum wavelength (Angstroms)		float
;
; OUTPUTS:  
;	Files containing image cubes in Rayleighs, and ancillary data, see WRITECUBEFILE
;         Note that the image cube output is Rayleighs per bin not per unit wavelength
;         Currently using 0.4 Angstrom spectral bins
;       Plots and optional plot files, see PLOTIMAGES
;
; COMMON BLOCKS:
;	None
;
; PROCEDURE:
;	Reads netCDF input file containing global airglow distribution in the far ultraviolet
;       calculated by the GLOW model, which used an empirical or numerical model as input.
;	Calculates paths through atmosphere for limb and disk as observed from geostationary
;       orbit, e.g., by the GOLD instrument.
;	Calculates radiative transfer for O(5S) 135.6 doublet and guesstimates O(3S) 130.4 triplet.
;	Includes N 149.3 line
;	Optionally calculates LBH spectral distribution.
;       Interpolates atmospheric fields and airglow volume emission rates along paths.
;       Integrates volume emission rates, including absorption, to calculate column brighnesses.
;	Writes netCDF output file and makes plots, optionally writes plot files. 
;
; ROUTINES CALLED:
;	Application-specific subroutines:  INTERGRID, RTO5S, LBHSPEC, WRITECUBEFILE, PLOTIMAGES
;	I/O utilities:  READ_NETCDF, WRITE_NETCDF
;	Geometry utilities:  LENGTH, ANGLE, DOT, SZA, SUNCOR, CHAP
;	Coordinate transforms:  LLA_TO_ECR, ECR_TO_LLA, CHIPSPI_TO_THETAPHI
;	Color handling utilities:  COLORSPEC, COLORBAR, COLORWIND
;	Graphics utilities:  TVREAD (special version of regular IDL routine, from David Fanning)
;	IDL internal:  INTERPOL, MAP_SET, MAP_CONTINENTS, MAP_GRID, WRITE_GIF
;
; REFERENCES:
;	Solomon, S. C., "Observing system simulations for the GOLD mission,"
;	http://download.hao.ucar.edu/pub/stans/gold/pptx/GoldOssFeb2016.pptx
;
; MODIFICATION HISTORY:
;	v. 0.62     preliminary release for initial evaluation         Stan Solomon, 2/2016
;       v. 0.71     flexible wavelength intervals and spatial grid     SCS, 2/2017
;       v. 0.72     made NI 1493 triplet into two components: (1492.63+1492.82) and 1494.68
;
;+

; --------------------------------------------------------------------

; Specify input and output file paths:

amodel='MSIS/IRI'
gmodel='GLOW v. 0.982'
directory='/hao/aimdata/stans/data/glowout/msisiri/gold2018/day317/'
filecode='msis'

infilepath=directory+'glow.'+filecode
outfilepath=directory+'cube.'+filecode
plotfilepath=directory+'plot.'+filecode

; Specify time range to process, and set switches for spectrum, plots and output

nstart=40
nstop=40
nplots=2
calclbhspec=1
writecube=0
writeplots=0
displayplots=1

; Specify satellite location:

satlat = 0.
satlon =-47.5
satalt = 35786.                            ; km
sat=lla_to_ecr([satlat,satlon,satalt])
satlen = length(sat)                       ; km
satrad = satlen/6370.                      ; Re

; Set up image grid:

; Limb setup:

;nchi=100
;npsi=50
;chimin=+8.5
;chimax=+9.5
;psimin=0.
;psimax=+3.

; Disk setup:

 nchi=100
 npsi=100
 chimin=-10.
 chimax=+10.
 psimin=-10.
 psimax=+10.

chi=chimin+findgen(nchi)*(chimax-chimin)/nchi
psi=psimin+findgen(npsi)*(psimax-psimin)/npsi

; Set up wavelength grid:

nwave=800                                  ; number of wavelength bins
apb=0.4                                    ; Angstroms per bin
wavemin=1300.2                             ; minimum wavelength in Angstroms
wavemax=wavemin+(nwave-1)*apb              ; maximum wavelength in Angstroms
wave=wavemin+findgen(nwave)*apb            ; array of bin center wavelengths in Angstroms
waves=wave-apb/2.                          ; array of bin starting boundary wavelengths

; Find bins corresponding to important wavelengths:

w1302=value_locate(waves,1302.17)
w1304=value_locate(waves,1304.86)
w1306=value_locate(waves,1306.03)
w1352=value_locate(waves,1352.)
w1356=value_locate(waves,1355.60)
w1359=value_locate(waves,1358.51)
w1361=value_locate(waves,1361.)
w1370=value_locate(waves,1370.)
w1410=value_locate(waves,1410.)
w1493=value_locate(waves,1492.72)
w1495=value_locate(waves,1494.68)
w1528=value_locate(waves,1528.)
w1619=value_locate(waves,1619.)

; Read netCDF file containing O2 far-ultraviolet cross sections and interpolate to wavelength grid:

read_netcdf,'o2absxsectfuv.nc',a
xo2=interpol(a.o2xsect,a.wavelength,wave)

; Interpolate O2 cross sections for atomic line calculations:

xo2_1304=interpol(a.o2xsect,a.wavelength,1304.)
xo2_1356=interpol(a.o2xsect,a.wavelength,1356.)
xo2_1359=interpol(a.o2xsect,a.wavelength,1359.)
xo2_1493=interpol(a.o2xsect,a.wavelength,1493.)

; Average O2 cross sections for broad-band calculations:

avxo2=mean(xo2[where(wave ge 1370 and wave le 1620)])
avxo2s=mean(xo2[where(wave ge 1410 and wave le 1528)])
avxo230=mean(xo2[where(wave ge 1352 and wave le 1361)])

; Atomic oxygen cross sections for 1356 RT:

xoxo=2.5e-18
xox1356=xoxo * .75 / 2.
xox1359=xoxo * .15 / 2.


; Loop over number of files:

for ntime=nstart,nstop do begin

  print,'starting frame number ',ntime

  ; Read netCDF file containing output from GLOW:

  infile=infilepath+'.'+string(ntime,format='(i3.3)')+'.nc'
  read_netcdf,infile,d

  ; Obtain model parameters and set up arrays:

  nlon=n_elements(d.lon)
  nlat=n_elements(d.lat)
  nlev=n_elements(d.lev)
  dlon=360./nlon
  dlat=180./nlat
  lonmin=-180.
  lonmax= 180-dlon
  latmin= -90+dlat
  latmax=  90-dlat
  date=d.idate
  ut=d.ut
  f107a=d.f107a
  f107=d.f107
  ap=d.ap
  lon=d.lon
  lat=d.lat
  zcm=d.zzz
  zkm=zcm/1.e5
  ao=d.ao
  ao2=d.ao2
  an2=d.an2
  atn=d.atn

  ; Construct image cube:

  if ntime eq nstart then begin            ; calculate path coordinates first time only

    ; Transform image grid to ECR coordinates and calculate nadir observation angle alpha:

    chir=chi*!dpi/180.
    psir=psi*!dpi/180.
    chipsi_to_thetaphi,satlon*!dpi/180.,chir,psir,theta,phi
    alpha=dblarr(nchi,npsi)
    for j=0,npsi-1 do for i=0,nchi-1 do alpha[i,j]=acos(cos(chir[i])*cos(psir[j]))

    ; Calculate coordinates of path segments through the atmosphere:

    npts=100
    altmin= 90.
    altrefdisk=150.
    altmax=500.
    v1sphere=dblarr(3,nchi,npsi)
    v2sphere=dblarr(3,nchi,npsi)
    v3sphere=dblarr(3,nchi,npsi)
    v1cart=dblarr(3,nchi,npsi)
    v2cart=dblarr(3,nchi,npsi)
    v3cart=dblarr(3,nchi,npsi)
    pierce=dblarr(3,nchi,npsi)
    pexit=dblarr(3,nchi,npsi)
    path=dblarr(3,nchi,npsi)
    pref=fltarr(3,nchi,npsi)
    gref=fltarr(3,nchi,npsi)
    ezaref=fltarr(nchi,npsi)
    flag=intarr(nchi,npsi)
    lpath=fltarr(nchi,npsi)
    dpath=fltarr(nchi,npsi)
    cpath=fltarr(3,npts,nchi,npsi)
    gpath=fltarr(3,npts,nchi,npsi)
    altp1=intarr(npts,nchi,npsi)
    altp2=intarr(npts,nchi,npsi)
    altfac=fltarr(npts,nchi,npsi)

    ; Bootstrap Re at pierce point, reference point, and exit point, and iterate once:

    reinit=6357.3                   ; initial Re because psi will start near 81 degrees lat
    repierce=reinit
    repref=reinit
    repexit=reinit

    ; For each image location, find coords of pierce point, reference alt, and exit point:

    for j=0,npsi-1 do begin
      for i=0,nchi-1 do begin
        for iterate=0,1 do begin
          rmax=repierce+altmax
          rref=repref+altrefdisk
          rmin=repexit+altmin
          determax=rmax^2-(satlen*sin(alpha[i,j]))^2
          deterref=rref^2-(satlen*sin(alpha[i,j]))^2
          determin=rmin^2-(satlen*sin(alpha[i,j]))^2
          if determax le 0. then begin                  ; l.o.s. doesn't pass through the atmosphere
            flag[i,j]=0
          endif else begin                              ; l.o.s. passes through the atmosphere
            v1sphere[0,i,j]=satlen*cos(alpha[i,j])-sqrt(determax)
            v1sphere[1,i,j]=theta[j]
            v1sphere[2,i,j]=phi[i]
            v1cart[*,i,j]=sphere_to_cart(v1sphere[*,i,j])
            pierce[*,i,j]=sat+v1cart[*,i,j]
            if determin lt 0. then begin                ; l.o.s. passes through the limb
              flag[i,j]=1
              v2sphere[0,i,j]=satlen*cos(alpha[i,j])+sqrt(determax)
              v2sphere[1,i,j]=theta[j]
              v2sphere[2,i,j]=phi[i]
              v2cart[*,i,j]=sphere_to_cart(v2sphere[*,i,j])
              pexit[*,i,j]=sat+v2cart[*,i,j]
              pref[*,i,j]=(pierce[*,i,j]+pexit[*,i,j])/2.
            endif else begin                            ; l.o.s. passes through the disk
              flag[i,j]=2
              v2sphere[0,i,j]=satlen*cos(alpha[i,j])-sqrt(determin)
              v2sphere[1,i,j]=theta[j]
              v2sphere[2,i,j]=phi[i]
              v2cart[*,i,j]=sphere_to_cart(v2sphere[*,i,j])
              pexit[*,i,j]=sat+v2cart[*,i,j]
              v3sphere[0,i,j]=satlen*cos(alpha[i,j])-sqrt(deterref)
              v3sphere[1,i,j]=theta[j]
              v3sphere[2,i,j]=phi[i]
              v3cart[*,i,j]=sphere_to_cart(v3sphere[*,i,j])
              pref[*,i,j]=sat+v3cart[*,i,j]
            endelse

            ; update repierce, reref, and repexit:

            gpierce=ecr_to_lla(pierce[*,i,j])
            gpierce[2]=0.
            repierce=length(lla_to_ecr(gpierce))
            gpexit=ecr_to_lla(pexit[*,i,j])
            gpexit[2]=0.
            repexit=length(lla_to_ecr(gpexit))
            gpref=ecr_to_lla(pref[*,i,j])
            gpref[2]=0.
            repref=length(lla_to_ecr(gpref))
          endelse
        endfor
      endfor
    endfor

    ; Set flag locations:

    w0f=where(flag eq 0)
    w1f=where(flag eq 1)
    w2f=where(flag eq 2)

    ; Calculate observation zenith angle, geolocation, local solar time, and solar zenith angle
    ; at each reference point:

    for j=0,npsi-1 do begin
      for i=0,nchi-1 do begin
        ezaref[i,j]=angle(pref[*,i,j],sat-pref[*,i,j])*180./!pi
        gref[*,i,j]=ecr_to_lla(pref[*,i,j])
      endfor
    endfor
    ezaref[w0f]=0.
    gref[*,w0f]=0.
    latref=reform(gref[0,*,*])
    lonref=reform(gref[1,*,*])
    altref=reform(gref[2,*,*])
    lstref=(ut/3600.+lonref/15.+24.) mod 24.
    szaref=sza(date,ut,latref,lonref)

    ; For each image location, calculate coords of each point along path through atmosphere:

    fpath=findgen(npts)/npts
    for j=0,npsi-1 do begin
      for i=0,nchi-1 do begin
        if flag[i,j] gt 0 then begin
          path[*,i,j]=pexit[*,i,j]-pierce[*,i,j]
          lpath[i,j]=float(length(path[*,i,j]))
          dpath[i,j]=lpath[i,j]/float(npts+1)
          cpath[0,*,i,j]=float(pierce[0,i,j]+path[0,i,j]*fpath)
          cpath[1,*,i,j]=float(pierce[1,i,j]+path[1,i,j]*fpath)
          cpath[2,*,i,j]=float(pierce[2,i,j]+path[2,i,j]*fpath)
          gpath[*,*,i,j]=ecr_to_lla(cpath[*,*,i,j])                           ; km
        endif
      endfor
    endfor
  endif                 ; end of first-time conditional
 
  ; Find location of each point along path in the model grid, and calculate interpolation factors:

  print,'calculating interpolation coordinates'

  lonp1=fix(reform(value_locate(lon,gpath[1,*,*,*])) > 0)
  lonp2=(lonp1+1)<(nlon-2)
  lonfac2=reform(float((gpath[1,*,*,*]-lon[lonp1])/dlon))
  lonfac1=1.-lonfac2
  latp1=fix(reform(value_locate(lat,gpath[0,*,*,*])) > 0) 
  latp2=(latp1+1)<(nlat-2)
  latfac2=reform(float((gpath[0,*,*,*]-lat[latp1])/dlat))
  latfac1=1.-latfac2
  for j=0,npsi-1 do begin
    for i=0,nchi-1 do begin
      if flag[i,j] gt 0 then begin
      for l=0,npts-1 do begin
        alti1=value_locate(zkm[lonp1[l,i,j],latp1[l,i,j],*],gpath[2,l,i,j]) > 0
        alti1=alti1<(nlev-2)
        alti2=alti1+1
        alt1=zkm[lonp1[l,i,j],latp1[l,i,j],alti1]
        alt2=zkm[lonp1[l,i,j],latp2[l,i,j],alti1]
        alt3=zkm[lonp2[l,i,j],latp1[l,i,j],alti1]
        alt4=zkm[lonp2[l,i,j],latp2[l,i,j],alti1]
        altav1=alt1*lonfac1[l,i,j]*latfac1[l,i,j]+alt2*lonfac1[l,i,j]*latfac2[l,i,j] $
              +alt3*lonfac2[l,i,j]*latfac1[l,i,j]+alt4*lonfac2[l,i,j]*latfac2[l,i,j]
        alt1=zkm[lonp1[l,i,j],latp1[l,i,j],alti2]
        alt2=zkm[lonp1[l,i,j],latp2[l,i,j],alti2]
        alt3=zkm[lonp2[l,i,j],latp1[l,i,j],alti2]
        alt4=zkm[lonp2[l,i,j],latp2[l,i,j],alti2]
        altav2=alt1*lonfac1[l,i,j]*latfac1[l,i,j]+alt2*lonfac1[l,i,j]*latfac2[l,i,j] $
              +alt3*lonfac2[l,i,j]*latfac1[l,i,j]+alt4*lonfac2[l,i,j]*latfac2[l,i,j]
        altfac[l,i,j]=(gpath[2,l,i,j]-altav1)/(altav2-altav1)
        altp1[l,i,j]=alti1
        altp2[l,i,j]=alti2
      endfor
      endif
    endfor
  endfor

  ; Interpolate various model output fields at each point along path:

  intergrid,ao,lonp1,lonp2,latp1,latp2,altp1,altp2, $
            lonfac1,lonfac2,latfac1,latfac2,altfac,ox
  ox[*,w0f]=0.
  intergrid,ao2,lonp1,lonp2,latp1,latp2,altp1,altp2, $
            lonfac1,lonfac2,latfac1,latfac2,altfac,o2
  o2[*,w0f]=0.
  intergrid,an2,lonp1,lonp2,latp1,latp2,altp1,altp2, $
            lonfac1,lonfac2,latfac1,latfac2,altfac,n2
  n2[*,w0f]=0.
  intergrid,atn,lonp1,lonp2,latp1,latp2,altp1,altp2, $
            lonfac1,lonfac2,latfac1,latfac2,altfac,tn
  tn[*,w0f]=0.

  ; Calculate O and O2 slant column density above each path point:

  scdox=fltarr(npts,nchi,npsi)
  for l=0,npts-1 do scdox[l,*,*]=total(ox[0:l,*,*],1)*dpath*1.e5        ; km to cm
  scdo2=fltarr(npts,nchi,npsi)
  for l=0,npts-1 do scdo2[l,*,*]=total(o2[0:l,*,*],1)*dpath*1.e5        ; km to cm

  ; Calculate O(5S) doublet total final source function at 135.6 and 135.9 nm
  ; using monochromatic radiative transfer approximation, then calculate slant column brightness:

  print,'calculating 1356 RT'

  vero5s=reform(d.eta[*,*,*,12])
  rto5s,zcm,ao,ao2,atn,vero5s,tfsf1356,tfsf1359

  ; Calculate emergent intensity for lines and bands:

  intergrid,tfsf1356,lonp1,lonp2,latp1,latp2,altp1,altp2, $
              lonfac1,lonfac2,latfac1,latfac2,altfac,sver1356
  sver1356[*,w0f]=0.
  scb1356=total(sver1356*exp(-scdo2*xo2_1356-scdox*xox1356),1)*dpath*1.e5   ; km to cm
  scb1356=scb1356*1.e-6                                                              ; mR to R
  intergrid,tfsf1359,lonp1,lonp2,latp1,latp2,altp1,altp2, $
              lonfac1,lonfac2,latfac1,latfac2,altfac,sver1359
  sver1359[*,w0f]=0.
  scb1359=total(sver1359*exp(-scdo2*xo2_1359-scdox*xox1359),1)*dpath*1.e5   ; km to cm
  scb1359=scb1359*1.e-6                                                              ; mR to R
  scbo5s=scb1356+scb1359

  ; Approximate O(3S) triplet at 130.2 130.4 130.6 nm slant column brightness:

  vero3s=reform(d.eta[*,*,*,14])
  intergrid,vero3s,lonp1,lonp2,latp1,latp2,altp1,altp2, $
            lonfac1,lonfac2,latfac1,latfac2,altfac,svero3s
  svero3s[*,w0f]=0.
  scbo3s=total(svero3s*exp(-scdo2*xo2_1304),1)*dpath*1.e5                 ; km to cm
  scbo3s=scbo3s*1.e-6                                                              ; mR to R
  diskfac=3.6/(chap(ezaref,6520.e5,50e5))^0.90
  limbfac=exp(1.47718-0.0344111*altref+0.00013677*altref^2-1.21648e-7*altref^3)
  scbo3s[w2f]=scbo3s[w2f]*diskfac[w2f]
  scbo3s[w1f]=scbo3s[w1f]*limbfac[w1f]
  scbo3s=smooth(scbo3s,5)
  scb1302=scbo3s*0.30
  scb1304=scbo3s*0.35
  scb1306=scbo3s*0.35

  ; Calculate N 1493 line slant column brightness:

  ver1493=reform(d.eta[*,*,*,13])
  intergrid,ver1493,lonp1,lonp2,latp1,latp2,altp1,altp2, $
              lonfac1,lonfac2,latfac1,latfac2,altfac,sver1493
  sver1493[*,w0f]=0.
  scb1493=total(sver1493*exp(-scdo2*xo2_1493),1)*dpath*1.e5                          ; km to cm
  scb1493=scb1493*1.e-6                                                              ; mR to R

  ; Estimate total LBH volume emission & column brightness 137-160, 141-153, (3,0) band at 135.4 nm:

  spectfac=0.40                               ; estimated ratio LBH 137-160 nm to total band system
  shortfac=0.23                               ; estimated ratio LBH 141-153 nm to total band system
  threefac=0.07                               ; estimated ratio LBH (3,0) band to total band system
  lbhsys=reform(d.eta[*,*,*,11])
  intergrid,lbhsys,lonp1,lonp2,latp1,latp2,altp1,altp2, $
              lonfac1,lonfac2,latfac1,latfac2,altfac,sverlbh
  sverlbh[*,w0f]=0.
  scblbht=total(spectfac*sverlbh*exp(-scdo2*avxo2),1)*dpath*1.e5                     ; km to cm
  scblbht=scblbht*1.e-6                                                              ; mR to R
  scblbhs=total(shortfac*sverlbh*exp(-scdo2*avxo2s),1)*dpath*1.e5                    ; km to cm
  scblbhs=scblbhs*1.e-6                                                              ; mR to R
  scblbh30=total(threefac*sverlbh*exp(-scdo2*avxo230),1)*dpath*1.e5                  ; km to cm
  scblbh30=scblbh30*1.e-6                                                            ; mR to R

  ; Calculate 1356/LBHs ratio, and set slant column brightness spectral array to zero:

  rat=(scbo5s+scblbh30)/(scblbhs+scb1493)
  rat[where(scblbht lt 0.1)]=0.
  scb=0.

  ; Calculate effective temperature and temperature at reference altitude:

  teff=total(tn*spectfac*sverlbh*exp(-scdo2*avxo2),1)*dpath*1.e5/(scblbht*1e6)
  teff[where(scblbht lt 0.1)]=0.
  tref=fltarr(nchi,npsi)
  for j=0,npsi-1 do begin
    for i=0,nchi-1 do begin
      if flag[i,j] eq 1 then begin
        tref[i,j]=tn[npts/2,i,j]
      endif
      if flag[i,j] eq 2 then begin
        tpath=tn[*,i,j]
        zpath=gpath[2,*,i,j]
        tref[i,j]=interpol(tpath,zpath,altref[i,j])
      endif
    endfor
  endfor

  ; Optionally calculate LBH synthetic spectrum using N2(a)v' distribution from GLOW output:

  if calclbhspec then begin

    print,'calculating LBH spectrum volume emission rates'

    spec=fltarr(nlon,nlat,nlev,nwave)
    for k=0,nlev-1 do begin
      for j=0,nlat-1 do begin
        for i=0,nlon-1 do begin
          t=atn[i,j,k]
          vplbh=reform(d.lbh[i,j,k,*])
          lbhspec,wavemin,wavemax,t,vplbh,wavelbh,spectrum
          spec[i,j,k,*]=spectrum
        endfor
      endfor
    endfor

    ; Calculate LBH spectrum slant column brightnesses:

    print,'calculating LBH spectrum slant column brightnesses'

    scblbh=fltarr(nchi,npsi,nwave)
    for w=0,nwave-1 do begin
      specint=reform(spec[*,*,*,w])
      intergrid,specint,lonp1,lonp2,latp1,latp2,altp1,altp2, $
                lonfac1,lonfac2,latfac1,latfac2,altfac,sver
      sver[*,w0f]=0.
      scblbh[*,*,w]=total(sver*exp(-scdo2*xo2[w]),1)*dpath*1.e5                       ; km to cm
    endfor
    scblbh=scblbh*1.e-6                                                               ; mR to R 
    scb=scblbh

    ; Calculate LBH wavelength integrals, and add atomic lines to spectrum:

    scblbht=total(scb[*,*,w1370:w1619],3)
    scblbhs=total(scb[*,*,w1410:w1528],3)
    scblbh30=total(scb[*,*,w1352:w1361],3)
    scb[*,*,w1356]=scb[*,*,w1356]+scb1356
    scb[*,*,w1359]=scb[*,*,w1359]+scb1359
    scb[*,*,w1493]=scb[*,*,w1493]+scb1493*2./3.
    scb[*,*,w1495]=scb[*,*,w1495]+scb1493/3.
    scb[*,*,w1302]=scb[*,*,w1302]+scb1302
    scb[*,*,w1304]=scb[*,*,w1304]+scb1304
    scb[*,*,w1306]=scb[*,*,w1306]+scb1306

    ; Calculate ratio of "1356" to "LBH short":

    rat=(scbo5s+scblbh30)/(scblbhs+scb1493)
    rat[where(scblbht lt 0.1)]=0.

  endif

  ; Write output file if requested:

  if writecube then writecubefile, $
     ntime,outfilepath,amodel,gmodel,date,ut,f107,f107a,ap,chi,psi,wave,scb, $
     scbo5s,scblbht,scblbhs,scblbh30,altref,latref,lonref,lstref,szaref,ezaref,tref,teff,flag

  ; Make plots if requested:

  if displayplots then nodisplay=0 else nodisplay=1
  if (writeplots or displayplots) then plotimages, $
     ntime,nplots,plotfilepath,date,ut,amodel,gmodel,satlat,satlon,satrad, $
     scbo5s,scblbhs,scblbh30,rat,teff,writeplots,nodisplay

endfor          ; bottom of time loop

end