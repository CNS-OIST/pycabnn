
(import scheme chicken)
(use typeclass kd-tree)
(use srfi-1 srfi-4 srfi-13 srfi-63 datatype getopt-long mpi)
(include "mathh-constants.scm")
(require-library data-structures files posix irregex mathh extras random-mtzig bvsp-spline parametric-curve)
(import foreign
	(only irregex string->irregex irregex-match)
	(only files make-pathname)
	(only posix glob)
	(only data-structures ->string alist-ref compose identity string-split)
	(only extras fprintf read-line read-lines pp)
	(only mathh cosh tanh log10)
	(only kd-tree <KdTree> <Point> Point3d KdTree3d make-point point?
	      kd-tree? kd-tree-for-each kd-tree-fold-right kd-tree-map
	      kd-tree->list kd-tree->list* kd-tree-empty? 
	      kd-tree-size kd-tree-min-index kd-tree-max-index
	      kd-tree-is-valid? kd-tree-all-subtrees-are-valid?
	      )
	(prefix bvsp-spline bvsp-spline:)
	(prefix parametric-curve pcurve:)
	(prefix random-mtzig random-mtzig:)
	)

(include "parser.scm")

; sign function
(define (sign x) (if (negative? x) -1.0 1.0))

; This returns 0 and will set make-parameter to 0- Or so. https://wiki.call-cc.org/man/4/Parameters#make-parameter
(define brep-verbose (make-parameter 0))

; Print function. fstr is a string that explains the values in args. 
; After that all values from args will be printed one after the other.
; However all of this only happens when brep-verbose is non- negative.
; As I got it, dotted pairs and lists are a thing in lisp. If you encounter a .(...), you can read that  as if the dot and the parentheses were removed.
; In this case, the input would kinda translate to *args in python.
(define (d fstr . args)
  (let ([port (current-error-port)])
    (if (positive? (brep-verbose)) 
	(begin (apply fprintf port fstr args)
	       (flush-output port) ) )))

; Check if this vector is empty
(define (f64vector-empty? x) (zero? (f64vector-length x)))

; http://chicken.kitten-technologies.co.uk/cache/4/spatial-trees/1.1/kd-tree.scm
; It seems as if the  <> brackets are defining an instance, i.e. some kind of object on a class. 
; ??
(import-instance (<KdTree> KdTree3d)
		 (<Point> Point3d))

; Gives back a random integer froma uniform distribution that is in the range [low high].
; Exchanging low and high as arguments should be no problem.
; st is the state vector, can be generated using (random-mtzig:init seed)
(define (random-uniform low high st)
  (let ((rlo (if (< low high) low high)) ;rlo gives back the real lower one,
	(rhi (if (< low high) high low))) ;rhi gives the real higher one.  
    (let ((delta (+ 1 (- rhi rlo))) ;delta = rhi- rlo + 1
	  (v (random-mtzig:randu! st)))random-seed ; randu! gives a sample frm a uniform distribution in interval (0,1)
      (+ rlo (floor (* delta v))) ; rlo* floor(v * delta)
      ))
  )

; Gives back a random number from a normal distribution with mean and sdev as defined
(define (random-normal mean sdev st)
   (+ mean (* sdev (random-mtzig:randn! st))))


;; convenience procedure to access to results of kd-tree-nearest-neighbor queries
; Can be used as a description for the datatype kdnn
(define (kdnn-point x) (cadr x))
(define (kdnn-distance x) (caddr x))
(define (kdnn-index x) (caar x))
(define (kdnn-parent-index x) (car (cadar x)))
(define (kdnn-parent-distance x) (cadr (cadar x)))


; It seems as if f64vectors are used here to represent points in space.
; The point-> list function thus changes the datatype of f64v to list.
(define point->list f64vector->list)



(define-record-type pointset (make-pointset prefix id points boundary)
  pointset? 
  (prefix pointset-prefix)
  (id pointset-id)
  (points pointset-points)
  (boundary pointset-boundary)
  )


(define-record-type bounds
  (make-bounds top left bottom right)
  bounds?
  (top       bounds-top )
  (left      bounds-left )
  (bottom    bounds-bottom )
  (right     bounds-right )
  )


(define-record-type genpoint (make-genpoint coords parent-index parent-distance segment)
  genpoint? 
  (coords genpoint-coords)
  (parent-index genpoint-parent-index)
  (parent-distance genpoint-parent-distance)
  (segment genpoint-segment)
  )


(define-record-type cell (make-cell ty index origin sections)
  cell? 
  (ty cell-type)
  (index cell-index)
  (origin cell-origin)
  (sections cell-sections)
  )


; (cell-sections cell) returns an alist that contains pairs (name-of-cell-section cell-section) for a given cell.
; As I get it so far, an alist contains pairs, the first one is some kind of denominator, and the second element
; is another object (like a python dict.)
; Thus, this function gives back a certain cell-section of a given cell.
(define (cell-section-ref name cell)
  (alist-ref name (cell-sections cell)))


; Seems to iterate through the points and coordinates and print them...
(define (write-sections section-name cells)
  (lambda (out)
    
    (for-each ; for-ech loops iterate through the list from left to right, do things, but discard the return value. Map keeps it.
     (lambda (gx) ; cell loop

       (for-each
        (lambda (section) ; section loop
          (for-each 
           (lambda (gd) ; apparently a section consists of several genpoints
             (let ((p (genpoint-coords gd))) ; This seems to iterate through the coords
               (fprintf out "~A ~A ~A "
                        (coord 0 p)
                        (coord 1 p)
                        (coord 2 p))))
           (cdr section)))
        (cell-section-ref section-name gx))
       (fprintf out "~%"))

     cells)))


; The #!key argument means that there is an even number of elements expected, which will be seperated into pairs of keyword and parameter.
(define (cells-sections->kd-tree cells section-name #!key
                                 (make-value
                                  (lambda (i v) 
                                    (list (genpoint-parent-index v)
                                          (genpoint-parent-distance v))))
                                 (make-point
                                  (lambda (v) (genpoint-coords v))))
  (let ((t 
	 (let recur ((cells cells) (points '()))
	   (if (null? cells)
	       (list->kd-tree
		points
		make-value: make-value
		make-point: make-point)
	       (let ((cell (car cells)))
		 (recur (cdr cells) 
                        (let inner ((sections (append (cell-section-ref section-name cell)))
                                    (points points))
                          (if (null? sections) points
                              (inner
                               (cdr sections)
                               (append (cdr (car sections)) points))
                              ))
                        ))
	       ))
	 ))
    t))

(define (sections->kd-tree cells #!key
                          (make-value
                           (lambda (i v) 
                             (list (genpoint-parent-index v)
                                   (genpoint-parent-distance v))))
                          (make-point
                           (lambda (v) (genpoint-coords v))))
  (let ((t 
	 (let recur ((cells cells) (points '()))
	   (if (null? cells) ; when you have worked through all cells
	       (list->kd-tree  ; This is a predefined function from the kd-package
		points
		make-value: make-value
		make-point: make-point) ; else
               (let ((sections (car cells)))
                 (let inner ((sections sections) (points points))
                   (if (null? sections)
                       (recur (cdr cells) points) ; when all sections of this cell have been treated, go on wwith next cell
                       (let ((section (car sections))) ; else take next section
                         (inner (cdr sections)  
                                (append (cdr (cadr section)) points)) ;append some element from the section to the list of points
                         ))
                   ))
               ))
         ))
    t))

; ? no clue what the ((t t)) means.
(define (cells-origins->kd-tree cells)
  (let ((t 
	 (let recur ((cells cells) (points '()))
	   (if (null? cells)
	       (list->kd-tree
		points
		make-value: (lambda (i v) 
			      (list (genpoint-parent-index v)
				    (genpoint-parent-distance v)))
		make-point: (lambda (v) (genpoint-coords v))
		)
	       (let ((cell (car cells)))
		 (recur (cdr cells) 
			(cons (make-genpoint (cell-origin cell) (cell-index cell) 0. 0)
                              points)))
	       ))
	 ))
    t))



(define (make-line-segment x y z dx dy dz) 
  (let ((c (pcurve:line-segment 3 (list dx dy dz))))
    (pcurve:translate-curve 
     (list x y z)
     (pcurve:line-segment 3 (list dx dy dz)))
    ))


;; auxiliary function to create segment points
(define (make-segment si np sp xyz)
  (list-tabulate 
   np
   (lambda (i) (let* ((pi (+ sp i))
                      (p (make-point 
                           (f64vector-ref (car xyz) pi)
                           (f64vector-ref (cadr xyz) pi)
                           (f64vector-ref (caddr xyz) pi))))
                 (list si p)
                 ))
   ))


;;
;; Creates a process of the given segments and number of points per
;; segment from the given curve.
;;
(define (make-segmented-process c ns np)
  (let ((xyz (pcurve:iterate-curve c (* ns np))))
    (let recur ((si ns) (ax '()))
      (if (positive? si)
          (let ((sp (* (- si 1) np)))
            (recur (- si 1) (append (make-segment si np sp xyz) ax)))
          ax)
      ))
  )

;;
;; Non-segmented process
;;
(define (make-process c np)
  (let ((xyz (pcurve:iterate-curve c np)))
    (list-tabulate 
     np
     (lambda (i) 
       (make-point 
        (f64vector-ref (car xyz) i)
        (f64vector-ref (cadr xyz) i)
        (f64vector-ref (caddr xyz) i)))
     ))
  )


;;
;; Creates a named section containing points from the given segmented processes.
;;
(define (make-segmented-section gid p label ps)
  `(,label . 
           ,(fold (lambda (i+proc ax)
                    (let ((seci (car i+proc)) 
                          (proc (cadr i+proc)))
                      (cons 
                       `(,seci . 
                               ,(map (lambda (sp)
                                       (let ((segi (car sp))
                                             (dpoint (cadr sp)))
                                         (let ((soma-distance (sqrt (dist2 p dpoint))))
                                           (make-genpoint dpoint gid soma-distance segi))
                                         ))
                                     proc))
                       ax)
                      ))
                  '() ps)
    ))

;;
;; Non-segmented named section
;;
(define (make-section gid p label ps)
  `(,label . 
           ,(fold (lambda (i+proc ax)
                    (let* ((seci (car i+proc)) 
                           (proc (cadr i+proc))
                           (pts (map (lambda (dpoint)
                                       (let ((soma-distance (sqrt (dist2 p dpoint))))
                                         (make-genpoint dpoint gid soma-distance #f)))
                                     proc)))
                      (d "make-section: gid = ~A pts = ~A~%" gid proc)
                      (cons `(,seci . ,pts) ax)
                      ))
                  '() ps)
    ))



(define (make-gen-kd-tree data #!key (threshold 0.0))
  
  (define (update-bb p x-min y-min z-min x-max y-max z-max) ; adapt the min and max values in all dimensions to the new point
    (let ((x (coord 0 p)) (y (coord 1 p)) (z (coord 2 p)))
      (if (< x (x-min)) (x-min x))
      (if (< y (y-min)) (y-min y))
      (if (< z (z-min)) (z-min z))
      (if (> x (x-max)) (x-max x))
      (if (> y (y-max)) (y-max y))
      (if (> z (z-max)) (z-max z))
      ))


  (let* (
	  (t (list->kd-tree
	     (filter (lambda (x) (<= threshold (cdr x))) data)
	     make-value: (lambda (i v) (cdr v))
	     make-point: (lambda (v) (apply make-point (car v)))
	     offset: (get-layer-object-index-floor)
	     ))
	 (n (kd-tree-size t))
	 (x-min (make-parameter +inf.0))
	 (y-min (make-parameter +inf.0))
	 (z-min (make-parameter +inf.0))
	 (x-max (make-parameter -inf.0))
	 (y-max (make-parameter -inf.0))
	 (z-max (make-parameter -inf.0))
	 (bb (begin (kd-tree-for-each
		     (lambda (p) (update-bb p x-min y-min z-min
					    x-max y-max z-max))
		     t)
		    (list (x-min) (y-min) (z-min) (x-max) (y-max) (z-max))))
	 )

    (cons bb t)

    ))




(define (genpoint-projection prefix my-comm myrank size cells fibers zone cell-start fiber-start)

  (d "rank ~A: zone = ~A~%" myrank zone)

  (fold (lambda (cell ax)
          
          (let* ((i    (+ cell-start (car cell)))
                 (root (modulo i size))
                 (sections (cadr cell)))
            
            (fold 
             
             (lambda (sec ax)
               
               (let ((seci (car sec))
                     (gxs  (cdr sec)))
                 
                 (let ((query-data
                        (fold 
                         (lambda (gd ax)
                           (d "rank ~A: querying point ~A (~A)~%" 
                              myrank (genpoint-coords gd) 
                              (genpoint-parent-index gd))
                           (fold
                            (lambda (x ax) 
                              (let (
                                    (source (car x))
                                    (target i)
                                    (distance (cadr x))
                                    (segi (genpoint-segment gd))
                                    )
                                (append (list source target distance segi seci) ax)
                                ))
                            ax
                            (delete-duplicates
                             (map (lambda (x) 
                                    (d "rank ~A: query result = ~A (~A) (~A) ~%" 
                                       myrank (kdnn-point x) (kdnn-distance x) (kdnn-parent-index x))
                                    (list (+ fiber-start (kdnn-parent-index x))
                                          (+ (kdnn-distance x) (genpoint-parent-distance gd)  (kdnn-parent-distance x))
                                          ))
                                  (kd-tree-near-neighbors* fibers zone (genpoint-coords gd)))
                             (lambda (u v)
                               (= (car u) (car v)))
                             )
                            ))
                         '() gxs)))
                   
                   (let* ((res0 (MPI:gatherv-f64vector (list->f64vector query-data) root my-comm)) ;the MPI gatherv command makes all processes send their data of variable length. 
                          
                          (res1 (or (and (= myrank root) (filter (lambda (x) (not (f64vector-empty? x))) res0)) '())))
                     
                     (append res1 ax))
                   
                   ))
               )
             ax sections)
            ))
        '() cells ))


	  

(define (point-projection prefix my-comm myrank size pts fibers zone point-start nn-filter)

  (fold (lambda (px ax)
    
    (d "~A: rank ~A: px = ~A~%"  prefix myrank px)

    (let* (
      (i (+ point-start (car px)))
      (root (modulo i size))
      (dd (d "~A: rank ~A: querying point ~A (root ~A)~%" prefix myrank px root))
      (query-data
        (fold 
          (lambda (pd ax)
            (fold
              (lambda (x ax) 
                (let (
                  (source (car x))
                  (target i)
                  (distance (cadr x)))
                  (if (and (> distance  0.) (not (= source target)))
                    (append (list source target distance) ax)
                    ax)
                ))
          ax
          (delete-duplicates
            (map (lambda (x) 
              (let (
                (res 
                  (list 
                    (car (cadar x)) 
                    (+ (caddr x) (cadr (cadar x))))))

                (d "~A: axon: x = ~A res = ~A~%" prefix x res)
                res))
            (nn-filter pd (kd-tree-near-neighbors* fibers zone pd))
            )
           
           (lambda (u v) (= (car u) (car v)))
           )
          ))
        '() (list (cadr px))))
        )

      (let* (
        (res0 (MPI:gatherv-f64vector (list->f64vector query-data) root my-comm))
        (res1 
          (or 
            (and 
              (= myrank root) 
              (filter (lambda (x) (not (f64vector-empty? x))) res0)) 
            '())))
      (append res1 ax))
      
    ))

  '() pts
))
            



(define bounds-empty (make-bounds -inf.0 +inf.0 +inf.0 -inf.0))


(define (bounds-translate b dx dy)
  (make-bounds (+ dy (bounds-top b))
	       (+ dx (bounds-left b))
	       (+ dy (bounds-bottom b))
	       (+ dx (bounds-right b))))


(define (bounds-add b p)
  (make-bounds (fpmax (coord 1  p) (bounds-top b))
	       (fpmin (coord 0  p) (bounds-left b))
	       (fpmin (coord 1  p) (bounds-bottom b))
	       (fpmax (coord 0  p) (bounds-right b))))


(define-datatype layer-boundary layer-boundary?
  (Bounds (b bounds?))
  (BoundsXZ (b bounds?) (n integer?) (k integer?) (x f64vector?) (y f64vector?) (d f64vector?) (d2 f64vector?))
  (BoundsYZ (b bounds?) (n integer?) (k integer?) (x f64vector?) (y f64vector?) (d f64vector?) (d2 f64vector?))
  )


(define (layer-boundary-bounds b)
  (cases layer-boundary b
	 (Bounds (b) b)
	 (BoundsXZ (b n k x y d d2) b)
	 (BoundsYZ (b n k x y d d2) b)))


(define (boundary-z-extent-function boundary)
  (cases layer-boundary boundary
	 (Bounds (b) 
		 (lambda (x y) 0.))
	 (BoundsXZ (b n k x y d d2) 
		   (lambda (xp yp) 
		     (let-values (((y0tab y1tab y2tab res)
				   (bvsp-spline:evaluate n k x y d d2 (f64vector xp) 0)))
		       (f64vector-ref y0tab 0))))
	 (BoundsYZ (b n k x y d d2) 
		   (lambda (xp yp) 
		     (let-values (((y0tab y1tab y2tab res)
				   (bvsp-spline:evaluate n k x y d d2 (f64vector yp) 0)))
		       (f64vector-ref y0tab 0))))
	 ))


(define (point2d-rejection boundary)
  (let ((top    (bounds-top boundary))
	(bottom (bounds-bottom boundary))
	(left   (bounds-left boundary))
	(right  (bounds-right boundary)))
    (lambda (p)
      (let ((x (coord 0 p)) (y (coord 1 p)))
	(and (fp> x left)  (fp< x right) (fp> y bottom) (fp< y top) p)))
    ))


(define (compute-point3d p zu z-extent-function)
  (let* ((x (coord 0 p))
	 (y (coord 1 p))
	 (z-extent (z-extent-function x y)))
    (if (zero? zu)
	(make-point x y zu)
	(make-point x y (fp* zu z-extent)))
    ))


(define (cluster-point-rejection p cluster-pts mean-distance)
  (let ((D (* 4 mean-distance mean-distance))
        (nn (kd-tree-nearest-neighbor cluster-pts p)))
    (and (< (dist2 p nn) D) p)))



(define (XZAxis n k x-points z-points boundary)
  
  (if (not (>= n 3)) 
      (error 'generate-boundary "XZAxis requires at least 3 interpolation points (n >= 3)"))
               
  (cases layer-boundary boundary
         (Bounds (b)  
                 (let-values (((d d2 constr errc diagn)
                               (bvsp-spline:compute n k x-points z-points)))
                   
                   (if (not (zero? errc)) 
                       (error 'generate-boundary "error in constructing spline from boundary points" errc))
                   
                   (BoundsXZ b n k x-points z-points d d2)))
         
         (else (error 'generate-boundary "boundary argument to XZAxis is already a pseudo-3D boundary")))
  )


(define (Grid x-spacing y-spacing z-spacing boundary)

  (let* (
         (xybounds  (cases layer-boundary boundary
                           (Bounds (b) b)
                           (BoundsXZ (b n k x y d d2) b)
                           (BoundsYZ (b n k x y d d2) b)))
         (x-extent   (- (bounds-right xybounds) (bounds-left xybounds)))
         (y-extent   (- (bounds-top xybounds) (bounds-bottom xybounds)))
         (z-extent-function
          (boundary-z-extent-function boundary))
         (compute-grid-points3d
          (lambda (p z-spacing z-extent-function)
            (let* ((x (coord 0 p))
                   (y (coord 1 p))
                   (z-extent (z-extent-function x y)))
              (let recur ((points '()) (z (/ z-spacing 2.)))
                (if (> z z-extent)
                    points
                    (recur (cons (make-point x y z) points) (+ z z-spacing)))
                ))
            ))
         )
    
    (d "Grid: x-spacing = ~A~%" x-spacing)
    (d "Grid: y-spacing = ~A~%" y-spacing)
    (d "Grid: z-spacing = ~A~%" z-spacing)
    
    (d "Grid: x-extent = ~A~%" x-extent)
    (d "Grid: y-extent = ~A~%" y-extent)
    
    (let ((x-points (let recur ((points '()) (x (/ x-spacing 2.)))
                      (if (> x x-extent)
                          (list->f64vector (reverse points))
                          (recur (cons x points) (+ x x-spacing)))))
          (y-points (let recur ((points '()) (y (/ y-spacing 2.)))
                      (if (> y y-extent)
                          (list->f64vector (reverse points))
                          (recur (cons y points) (+ y y-spacing)))))
          )
      
      (let ((nx (f64vector-length x-points))
            (ny (f64vector-length y-points))
            )
        
        (let recur ((i 0) (j 0) (ax '()))
          (if (< i nx)
              (let ((x (f64vector-ref x-points i)))
                (if (< j ny)
                    (let* ((y   (f64vector-ref y-points j))
                           (p   (make-point x y))
                           (p3ds (if (zero? z-spacing)
                                     (list (make-point x y 0.0))
                                     (compute-grid-points3d p z-spacing z-extent-function)))
                           )
                      (recur i (+ 1 j) (if p3ds (append p3ds ax) ax)))
                    (recur (+ 1 i) 0 ax)))
              (let* ((t (list->kd-tree ax))
                     (n (kd-tree-size t)))
                (list t boundary)
                ))
        )))
  ))


(define (UniformRandomPointProcess n x-seed y-seed boundary)

  (let* (
	 (xybounds  (cases layer-boundary boundary
			   (Bounds (b) b)
			   (BoundsXZ (b n k x y d d2) b)
			   (BoundsYZ (b n k x y d d2) b)))
	 (x-extent   (- (bounds-right xybounds) (bounds-left xybounds)))
	 (y-extent   (- (bounds-top xybounds) (bounds-bottom xybounds)))
	 (z-extent-function (boundary-z-extent-function boundary))
	 )

    (let ((x-points (random-mtzig:f64vector-randu! n (random-mtzig:init x-seed)))
	  (y-points (random-mtzig:f64vector-randu! n (random-mtzig:init y-seed)))
	  (z-points (random-mtzig:f64vector-randu! n (random-mtzig:init (current-milliseconds)))))
      
      (let ((point-rejection1 (point2d-rejection xybounds)))
	
	(let recur ((i 0) (ax '()))
	  (if (< i n)
	      (let ((x (f64vector-ref x-points i))
		    (y (f64vector-ref y-points i))
		    (z (f64vector-ref z-points i)))
		(let ((p (make-point (fp* x x-extent) (fp* y y-extent))))
		  (let ((p3d 
			 (and (point-rejection1 p)
			      (compute-point3d p z z-extent-function))))
		    (recur (+ 1 i) (if p3d (cons p3d ax) ax)))))
	      (let* ((t (list->kd-tree ax))
		     (n (kd-tree-size t)))

		(list t boundary))))
	))
    ))


; THis function is done to render originating points for Glgi cells if this inforḿation is not provided as an input param
(define (ClusteredRandomPointProcess cluster-pts n mean-distance x-seed y-seed boundary)

  (let* (
	 (xybounds  (cases layer-boundary boundary
			   (Bounds (b) b)
			   (BoundsXZ (b n k x y d d2) b)
			   (BoundsYZ (b n k x y d d2) b)))
	 (x-extent   (- (bounds-right xybounds) (bounds-left xybounds)))
	 (y-extent   (- (bounds-top xybounds) (bounds-bottom xybounds)))
	 (z-extent-function (boundary-z-extent-function boundary))
	 )

    (let recur ((pts '()) (x-seed x-seed) (y-seed y-seed))

      (let ((x-points (random-mtzig:f64vector-randu! n (random-mtzig:init x-seed)))
            (y-points (random-mtzig:f64vector-randu! n (random-mtzig:init y-seed)))
            (z-points (random-mtzig:f64vector-randu! n (random-mtzig:init (current-milliseconds)))))
        
        (let ((point-rejection1 (point2d-rejection xybounds)))
          
          (let inner-recur ((j 0) (ax pts))
            (if (< j n)
                (let ((x (f64vector-ref x-points j))
                      (y (f64vector-ref y-points j))
                      (z (f64vector-ref z-points j)))
                  (let ((p (make-point (fp* x x-extent) (fp* y y-extent))))
                    (let ((p3d 
                           (and (point-rejection1 p)
                                (compute-point3d p z z-extent-function))))
                      (let ((pp (cluster-point-rejection p3d cluster-pts mean-distance)))
                        (inner-recur (+ 1 j)  (if pp (cons pp ax) ax))))))

                (if (< (length ax) n)
                    (recur ax (+ 1 x-seed) (+ 1 y-seed))
                    (let* ((t (list->kd-tree (take ax n)))
                           (n (kd-tree-size t)))
                      
                      (list t boundary))))
            ))
	))
    ))


(define (ParametricNeurites 
         Sections nNeurites nSegs nSegPts
         soma-points random-seed
         fn parameters)
  
  (let ((rngst (random-mtzig:init random-seed)))

    (reverse
     (car
      (fold
       (lambda (p ax)
         
         (d "ParametricNeurites: p = ~A~%" p)
         
         (let ((clst (car ax))
               (gid  (cadr ax)))
           
           (d "ParametricNeurites: gid = ~A~%" gid)
           
           (let (
                 (neurite-sections 
                  (car
                   (fold
                    (lambda (section nNeurites nSegs nSegPts parameter-set ax)
                      (let ((lst (car ax))
                            (secStart (cadr ax)))
                        (let ((args (append (list rngst gid p section nNeurites secStart nSegs nSegPts)
                                            parameter-set)))
                          (list (cons (apply fn args) lst) (+ secStart nNeurites))
                          ))
                      )
                    (list '() 0)
                    Sections nNeurites nSegs nSegPts
                    parameters)
                   ))

                 )
             
             (list (cons neurite-sections clst) (+ 1 gid))
             ))
         )
      '(() 0)
      soma-points))
     ))
  )


(define (ParametricNeurites*
         Sections nNeurites 
         soma-points random-seed
         fn parameters)
  
  (let ((rngst (random-mtzig:init random-seed)))

    (reverse
      (fold
       (lambda (gp ax)
         
         (d "ParametricNeurites*: gp = ~A~%" gp)
         (d "ParametricNeurites*: parameters = ~A~%" parameters)
         
         (let ((clst ax)
               (gid  (car gp))
               (p    (cadr gp)))
           
           (let (
                 (neurite-sections 
                  (map
                   
                   (lambda (section nNeurites parameter-set)
                     (let ((args (append (list rngst gid p section nNeurites)
                                         parameter-set)))
                       (apply fn args)
                       ))

                   Sections nNeurites parameters)
                  )
                 )
             
             (cons neurite-sections clst)
             ))
         )
      '()
      soma-points)
     ))
  )


(define (ConePerturbationNeurites 
         rngst gid p section nNeurites secStart nSegs nSegPts 
         theta-range theta-stdev h r)
  (let (
        (x (coord 0 p))
        (y (coord 1 p))
        (z (coord 2 p))
        (theta-min (car theta-range))
        (theta-max (cdr theta-range))
        )
    
    (d "ConePerturbationNeurites: theta-range = ~A theta-stdev = ~A h = ~A r = ~A~%" 
       theta-range theta-stdev h r)
    
    (make-segmented-section  

     gid p section
     
     (list-tabulate 

      nNeurites
                    
      (lambda (i)
        
        (let* (
               ;; rotation of neurite in X-Y plane
               (theta-degrees (if (even? i) 
                                  (random-normal theta-min theta-stdev rngst)
                                  (random-normal theta-max theta-stdev rngst)))
               (theta  (* (/ PI 180.) theta-degrees ))
               )
          
          (d "ConePerturbationNeurites: theta-degrees = ~A theta = ~A~%" theta-degrees theta)
          
          (let (
                (N-dX (* r (cos theta)))
                (N-dY (* r (sin theta)))
                (N-dZ h)
                )
            
            (d "ConePerturbationNeurites: r = ~A h = ~A N-dX = ~A N-dY = ~A N-dZ = ~A~%" r h N-dX N-dY N-dZ)
            
            (list (+ i secStart 1) (make-segmented-process (make-line-segment x y z N-dX N-dY N-dZ) nSegs nSegPts))
            )
          ))
      ))
    ))



(define (LinearPerturbationNeurites  
         rngst gid p section nNeurites secStart nSegs nSegPts 
         x-range y-range z-range)

  (let ((x (coord 0 p))
        (y (coord 1 p))
        (z (coord 2 p))
        (x-min (car x-range))
        (x-max (cdr x-range))
        (y-min (car y-range))
        (y-max (cdr y-range))
        (z-min (car z-range))
        (z-max (cdr z-range))
        )
    
    (make-segmented-section 
     
     gid p section  
     
     (list-tabulate 
      
      nNeurites
      
      (lambda (i)
        
        (let* (
               (N-dX (random-uniform x-min x-max rngst))
               (N-dY (random-uniform y-min y-max rngst))
               (N-dZ (random-uniform z-min z-max rngst))
               )

          (d "LinearPerturbationNeurites: x = ~A y = ~A z = ~A dX = ~A dY = ~A dZ = ~A~%" 
             x y z N-dX N-dY N-dZ)
          
          (list (+ 1 secStart i) (make-segmented-process (make-line-segment x y z N-dX N-dY N-dZ) nSegs nSegPts))
          )
        ))
     ))
  )


(define (LinearNeurites 
         rngst gid p section nNeurites 
         f-step f-length f-axis f-offset)
  
  (let* (
         (x (+ (coord 0 f-offset) (coord 0 p)))
         (y (+ (coord 1 f-offset) (coord 1 p)))
         (z (+ (coord 2 f-offset) (coord 2 p)))
         (n (inexact->exact (floor (/ f-length f-step))))
         )

    (d "LinearNeurites: gid = ~A p = ~A x = ~A y = ~A z = ~A f-step = ~A f-length = ~A f-axis = ~A f-offset = ~A~%" 
      gid p x y z f-step f-length f-axis f-offset )


    (make-section 
     
     gid p section  

     (list-tabulate 
      
      nNeurites
      
      (lambda (i)
        
        (let* ((nPts (inexact->exact (floor (/ f-length (abs f-step)))))
               (endp (let ((xyz (f64vector 0. 0. 0.)))
                       (f64vector-set! xyz f-axis (* f-length (sign f-step)))
                       xyz))
               (N-dX   (f64vector-ref endp 0))
               (N-dY   (f64vector-ref endp 1))
               (N-dZ   (f64vector-ref endp 2))
               )
          
          
          (d "LinearNeurites: endp = ~A~%" endp)
          
          (list (+ 1 i) (make-process (make-line-segment x y z N-dX N-dY N-dZ) nPts))
          
          ))
      
      ))
    
    ))
  
  
	
	

        
(define comment-pat (string->irregex "^#.*"))


(define (load-points-from-file filename)

  (let ((in (open-input-file filename)))
    
    (if (not in) (error 'load-points-from-file "file not found" filename))

    (let* ((lines
	    (filter (lambda (line) (not (irregex-match comment-pat line)))
		    (read-lines in)))

	   (point-data
	    (map (lambda (line) (apply make-point (map string->number (string-split line " \t"))))
		 lines))

	   (point-tree (list->kd-tree point-data))
	   )
      
      (list point-tree)
      
      ))
  )


(define (GC-GoC-connections GoC-Dendrites Fibers my-comm myrank size prefix goc-zone pf-start goc-start)

    (MPI:barrier my-comm)
	  
    (let ((my-results
           (map (lambda (dendrites)
                  (genpoint-projection prefix my-comm myrank size dendrites Fibers goc-zone goc-start pf-start))
                GoC-Dendrites)))
      
      (MPI:barrier my-comm)
      
      (call-with-output-file (sprintf "~Asources~A.dat"  prefix (if (> size 1) myrank ""))
	(lambda (out-sources)
	  (call-with-output-file (sprintf "~Atargets~A.dat"  prefix (if (> size 1) myrank ""))
	    (lambda (out-targets)
	      (call-with-output-file (sprintf "~Adistances~A.dat"  prefix (if (> size 1) myrank ""))
		(lambda (out-distances)
                  (call-with-output-file (sprintf "~Asegments~A.dat"  prefix (if (> size 1) myrank ""))
                    (lambda (out-segments)
                      (for-each 
                       (lambda (my-set)
                         (for-each 
                          (lambda (my-data)
                            (let* ((my-entry-len 5)
                                   (my-data-len (/ (f64vector-length my-data) my-entry-len)))
                              (d "rank ~A: length my-data = ~A~%" myrank my-data-len)
                              (let recur ((m 0))
                                (if (< m my-data-len)
                                    (let* (
                                           (my-entry-offset (* m my-entry-len))
                                           (source (inexact->exact (f64vector-ref my-data my-entry-offset)))
                                           (target (inexact->exact (f64vector-ref my-data (+ 1 my-entry-offset)))) ; 23 brackets
                                           (distance (f64vector-ref my-data (+ 2 my-entry-offset)))
                                           (segment (f64vector-ref my-data (+ 3 my-entry-offset)))
                                           (section (f64vector-ref my-data (+ 4 my-entry-offset)))
                                           )
                                      (fprintf out-sources   "~A~%" source)
                                      (fprintf out-targets   "~A~%" target)
                                      (fprintf out-distances "~A~%" distance)
                                      (fprintf out-segments  "~A ~A~%" segment section)
                                      (recur (+ 1 m)))))
                              ))
                          my-set))
                       my-results)
                      ))
                  ))
              ))
          ))
      ))



(define (GoC-GoC-connections GoC-Somas GoC-Axons my-comm myrank size 
                             prefix goc-zone goc-start
			     #!key (nn-filter (lambda (x nn) nn)))

    
    (MPI:barrier my-comm)
	  
    (let ((my-results (point-projection prefix my-comm myrank size GoC-Somas GoC-Axons goc-zone goc-start nn-filter)))
      
      (MPI:barrier my-comm)
      
      (call-with-output-file (sprintf "~Asources~A.dat"  prefix (if (> size 1) myrank ""))
	(lambda (out-sources)
	  (call-with-output-file (sprintf "~Atargets~A.dat"  prefix (if (> size 1) myrank ""))
	    (lambda (out-targets)
	      (call-with-output-file (sprintf "~Adistances~A.dat"  prefix (if (> size 1) myrank ""))
		(lambda (out-distances)
		  (for-each 
		   (lambda (my-data)
		     (let* ((my-entry-len 3)
			    (my-data-len (/ (f64vector-length my-data) my-entry-len)))
		       (d "~A: rank ~A: length my-data = ~A~%" prefix myrank my-data-len)
		       (let recur ((m 0))
			 (if (< m my-data-len)
			     (let ((source (inexact->exact (f64vector-ref my-data (* m my-entry-len))))
				   (target (inexact->exact (f64vector-ref my-data (+ 1 (* m my-entry-len)))))
				   (distance (f64vector-ref my-data (+ 2 (* m my-entry-len)))))
			       (fprintf out-sources "~A~%" source)
			       (fprintf out-targets "~A~%" target)
			       (fprintf out-distances "~A~%" distance)
			       (recur (+ 1 m)))))
		       ))
		   my-results)
		  ))
	      ))
	  ))
      ))


(define (GoC-GoC-gap-connections 
                    GoC-Somas my-GoC-Soma-tree my-comm myrank size
                    prefix goc-zone goc-start
                    #!key 
                    (nn-filter (lambda (x nn) nn)))

  (MPI:barrier my-comm)
	  
  (let ((my-results (point-projection prefix my-comm myrank size GoC-Somas my-GoC-Soma-tree goc-zone goc-start nn-filter)))
      
    (MPI:barrier my-comm)
      
    (call-with-output-file (sprintf "~Asources~A.dat"  prefix (if (> size 1) myrank ""))
      (lambda (out-sources)
	       (call-with-output-file (sprintf "~Atargets~A.dat"  prefix (if (> size 1) myrank ""))
	         (lambda (out-targets)
	      (call-with-output-file (sprintf "~Adistances~A.dat"  prefix (if (> size 1) myrank ""))
		(lambda (out-distances)
		  (for-each 
		   (lambda (my-data)
		     (let* ((my-entry-len 3)
			    (my-data-len (/ (f64vector-length my-data) my-entry-len)))
		       (d "~A: rank ~A: length my-data = ~A~%" prefix myrank my-data-len)
		       (let recur ((m 0))
			 (if (< m my-data-len)
			     (let ((source (inexact->exact (f64vector-ref my-data (* m my-entry-len))))
				   (target (inexact->exact (f64vector-ref my-data (+ 1 (* m my-entry-len)))))
				   (distance (f64vector-ref my-data (+ 2 (* m my-entry-len)))))
			       (fprintf out-sources "~A~%" source)
			       (fprintf out-targets "~A~%" target)
			       (fprintf out-distances "~A~%" distance)
			       (recur (+ 1 m)))))
		       ))
		   my-results)
		  ))
	      ))
	  ))
  ))


(define (GoC-GoC-distances GoC-Somas my-comm myrank size prefix goc-start)
    (let ((my-results
      (let recur ((gxs GoC-Somas) (my-results '()))  ; gxs consists of GoC-Somas at the beginning, my-results is an empty list.
        (if (null? gxs)   ; when all elements of gxs have been processed, my_results is returned in reversed order
		    (reverse my-results)
		    (let* (
          (gx (car gxs)) ; gx = first element of gxs and apparently the GoC-goC distances
          (_  (d "GoC-GoC distances: gx = ~A~%" gx)) ; if verbose, print those distances
          (gx-distances 
              (filter-map
                  (lambda (gy) 
                      (let ((py (cadr gy)))
                        (d  "GoC-GoC distances: px = ~A py = ~A~%" px py)
                        (and (not (= (car gx) (car gy))) 
                             (cons (car gy) (sqrt (dist2 px py)))
                             )))
                  GoC-Somas)))
        (recur (cdr gxs) (cons (list (car gx) gx-distances) my-results)))
        ))
     ))
                 

      (call-with-output-file (sprintf "~Adistances~A.dat"  prefix (if (> size 1) myrank ""))
        (lambda (out)
          (for-each 
           (lambda (my-data)
             (let ((i (car my-data))
                   (dists (cadr my-data)))
               (for-each
                (lambda (d) 
                  (fprintf out "~A ~A ~A~%" i (car d) (cdr d)))
                dists)))
           my-results)
          ))
      ))


(define opt-defaults ;default opts. Will be accessed by the function defopt x
  `(
    (pf-length . 100)
    (pf-step   . 30)
    (aa-length . 200)
    (aa-step   . 100)
    (z-extent . 150.)
    (y-extent . 300.)
    (x-extent . 1200.)
    (num-gc . 10000)
    (num-goc . 200)
    (mean-goc-distance . 50)
    (goc-grid-xstep . 200)
    (goc-grid-ystep . 500)
    (pf-goc-zone . 5)
    (aa-goc-zone . 5)
    (goc-goc-zone . 30)
    (goc-goc-gap-zone . 30)
    (goc-dendrites . 4)
    (goc-apical-nseg . 2)
    (goc-basolateral-nseg . 2)
    (goc-apical-nsegpts . 4)
    (goc-basolateral-nsegpts . 4)
    (goc-axons . 10)
    (goc-axonsegs . 1)
    (goc-axonpts . 2)
    (goc-axon-x-min . -200)
    (goc-axon-x-max . 200)
    (goc-axon-y-min . -200)
    (goc-axon-y-max . 200)
    (goc-axon-z-min . -30)
    (goc-axon-z-max . -200)
    (goc-apical-dendheight . 100.0)
    (goc-apical-radius . 100.0)
    (goc-basolateral-dendheight . 100.0)
    (goc-basolateral-radius . 100.0)
    (goc-theta-apical-min . 30)
    (goc-theta-apical-max . 60)
    (goc-theta-basolateral-min . 30)
    (goc-theta-basolateral-max . 60)
    (goc-theta-apical-stdev . 1)
    (goc-theta-basolateral-stdev . 1)
    ))


(define (load-config-file filename) ;loads config file -> config-alst
  (let ((in (open-input-file filename)))
    (if (not in) (error 'load-config-file "file not found" filename))
    (init-bindings)
    (let* ((lines (reverse (filter (lambda (x) (not (string-null? x))) (read-lines in))))
           (properties (filter-map
                        (lambda (line) 
                          (and (not (string-prefix? "//" line))
                               (let ((lst (string-split line "=")))
                                 (and (> (length lst) 1)
                                      (let ((key (string->symbol (string-trim-both (car lst) #\space)))
                                            (val (brep-eval-string (cadr lst))))
                                        (add-binding key val)
                                        (cons key val))))))
                        lines)))
      properties
    ))
  )

(define (defopt x) ;see opt-defaults list
  (alist-ref x opt-defaults))


; from the description of the getopt-long file: 
; each element of the grammar should have the following form:
; ((OPTION-NAME [DOCSTRING] (PROPERTY VALUE) ...)...)    
; for the property-value pairs required appears to be a value that must be given, 
; transformer applies the function in the argument to the input before storing it.      
(define opt-grammar
  `(
    (rng-seeds "Use the given seeds for random number generation"
               (value (required SEED-LIST)
                      (transformer ,(lambda (x) (map string->number
                                                     (string-split x ","))))))

    (mpi-split "perform MPI split operation"
               (value (required COLOR)
                      (transformer ,string->number)))
    
    (config-file "use the given hoc configuration file to obtain parameters"
                 (value (required FILENAME)))
    
    (x-extent "X-extent of patch"
               (value (required LENGTH)
                      (transformer ,string->number)))

    (y-extent "Y-extent of patch"
               (value (required LENGTH)
                      (transformer ,string->number)))

    (z-extent "Z-extent of patch"
               (value (required LENGTH)
                      (transformer ,string->number)))

    (pf-step "parallel fiber step size"
             (value (required LENGTH)
                    (transformer ,string->number)))

    (pf-length  "parallel fiber length"
		(value (required INDEX)
		       (transformer ,string->number))
		)

    (aa-step "ascending axon step size"
             (value (required LENGTH)
                    (transformer ,string->number)))

    (aa-length  "ascending axon length"
		(value (required INDEX)
		       (transformer ,string->number))
		)

    (pf-start  "starting index for parallel fiber points (default is 0)"
                (single-char #\i)
                (value (required INDEX)
                       (transformer ,string->number))
                )

    (num-gc     "number of granule cells and parallel fibers"
                (value (required INDEX)
                       (transformer ,string->number))
                )

    (gc-points  "load originating points for granule cells from given file (default is randomly generated)"
		(value (required FILENAME))
		)

    (gct-points  "load junction points for parallelf fibers from given file (default is offset from GC soma points)"
                 (value (required FILENAME))
                 )

    (num-goc     "number of Golgi cells"
                (value (required INDEX)
                       (transformer ,string->number))
                )

    (mean-goc-distance     "mean distance between Golgi cells and Golgi cell grid"
                           (value (required DISTANCE)
                                  (transformer ,string->number))
                           )

    (goc-theta-apical-min "min angle used to determine height of apical dendrite in z direction"
                          (value (required ANGLE)
                                 (transformer ,string->number)))
    (goc-theta-apical-max "max angle used to determine height of apical dendrite in z direction"
                          (value (required ANGLE)
                                 (transformer ,string->number)))
    (goc-theta-apical-stdev "stdev of angle used to determine height of apical dendrite in z direction"
                          (value (required ANGLE)
                                 (transformer ,string->number)))

    (goc-theta-basolateral-min "min angle used to determine height of basolateral dendrite in z direction"
                               (value (required ANGLE)
                                      (transformer ,string->number)))
    (goc-theta-basolateral-max "max angle used to determine height of basolateral dendrite in z direction"
                               (value (required ANGLE)
                                      (transformer ,string->number)))
    (goc-theta-basolateral-stdev "stdev of angle used to determine height of basolateral dendrite in z direction"
                               (value (required ANGLE)
                                      (transformer ,string->number)))

    (goc-apical-dendheight "height of Golgi cell apical dendritic cone"
                        (value (required LENGTH)
                               (transformer ,string->number)))

    (goc-basolateral-dendheight "height of Golgi cell basolateral dendritic cone"
                             (value (required LENGTH)
                                    (transformer ,string->number)))

    (goc-apical-radius "radius of Golgi cell apical dendrite cone"
                        (value (required LENGTH)
                               (transformer ,string->number)))

    (goc-basolateral-radius "radius of Golgi cell basolateral dendrite cone"
                             (value (required LENGTH)
                                    (transformer ,string->number)))

    (goc-dendrites "number of Golgi cell dendrites"
                   (value (required NUMBER)
                          (transformer ,string->number)))

    (goc-axons "number of Golgi cell axons"
                   (value (required NUMBER)
                          (transformer ,string->number)))

    (goc-axonsegs "number of Golgi cell axon segments"
                   (value (required NUMBER)
                          (transformer ,string->number)))

    (goc-axonpts "number of Golgi cell axon points"
                   (value (required NUMBER)
                          (transformer ,string->number)))

    (goc-axon-x-min "minimum extent of Golgi cell axons along X axis"
                    (value (required NUMBER)
                           (transformer ,(compose abs string->number))))

    (goc-axon-x-max "maximum extent of Golgi cell axons along X axis"
                    (value (required NUMBER)
                           (transformer ,(compose abs string->number))))


    (goc-axon-y-min "minimum extent of Golgi cell axons along Y axis"
                    (value (required NUMBER)
                           (transformer ,(compose abs string->number))))

    (goc-axon-y-max "maximum extent of Golgi cell axons along Y axis"
                    (value (required NUMBER)
                           (transformer ,(compose abs string->number))))

    (goc-axon-z-min "minimum extent of Golgi cell axons along Z axis"
                    (value (required NUMBER)
                           (transformer ,(compose abs string->number))))

    (goc-axon-z-max "maximum extent of Golgi cell axons along Z axis"
                    (value (required NUMBER)
                           (transformer ,(compose abs string->number))))

    (goc-apical-nseg "number of segments of GoC apical dendrite"
                        (value (required N)
                               (transformer ,string->number)))

    (goc-basolateral-nseg "number of segments of GoC basolateral dendrite"
                          (value (required N)
                               (transformer ,string->number)))

    (goc-apical-nsegpts "number of points per segments of GoC apical dendrite"
                        (value (required N)
                               (transformer ,string->number)))

    (goc-basolateral-nsegpts "number of points per segments of GoC basolateral dendrite"
                             (value (required N)
                                    (transformer ,string->number)))

    (pf-goc-zone "PF to Golgi cell connectivity zone"
		 (value (required RADIUS)
			(transformer ,string->number)))

    (aa-goc-zone "AA to Golgi cell connectivity zone"
		 (value (required RADIUS)
			(transformer ,string->number)))


    (goc-goc-zone "Golgi to Golgi cell connectivity zone"
               (value (required RADIUS)
                      (transformer ,string->number)))

    (goc-goc-gap-zone "Golgi to Golgi gap junction connectivity zone"
                      (value (required RADIUS)
                             (transformer ,string->number)))

    (goc-points  "load originating points for Golgi cells from given file (default is randomly generated)"
		(single-char #\g)
		(value (required FILENAME))
		)

    (goc-start  "starting index for Golgi cell points (default is 0)"
                (single-char #\j)
                (value (required INDEX)
                       (transformer ,string->number))
                )

    (save-aa    "save ascending axon points")
    (save-pf    "save parallel fiber points")

    (output    "specify output file prefix"
		(single-char #\o)
		(value (required PREFIX))
		)

    (verbose          "verbose mode"
                      (single-char #\v))


    (help         (single-char #\h))           

    ))

;; Process arguments and collate options and arguments into OPTIONS
;; alist, and operands (filenames) into OPERANDS.  You can handle
;; options as they are processed, or afterwards.

; getopt-long is a library that supports command-line parsing. It needs an opt-grammar as it is defned above.

(define opts    (getopt-long (command-line-arguments) opt-grammar)) 
(define opt     (make-option-dispatch opts opt-grammar))  ; okay, no clue where this comes from but it seems to be related to 9ML


;; Use args:usage to generate a formatted list of options (from OPTS),
;; suitable for embedding into help text.
(define (brep:usage)
  (print "Usage: " (car (argv)) " [options...] ")
  (newline)
  (print "The following options are recognized: ")
  (newline)
  (print (parameterize ((indent 5)) (usage opt-grammar)))
  (exit 1))













(define (main options operands)

  ; random seeds: get from file or use the ones that are defined as default
  (define rng-seeds (make-parameter (apply circular-list (or (options 'rng-seeds) (list 13 17 19 23 29 37)))))

  (define (get-rng-seeds . rest) ; get random seed
    (let-optionals rest ((n 1)) ; let-optionals binds optional arguments.
      (let ((v (take (rng-seeds) n)))
        (rng-seeds (drop (rng-seeds) n))
        v)))

  (if (options 'help) (brep:usage))

  (if (options 'verbose) (begin (brep-verbose 1)))

  (MPI:init) ; initializes MPI execution environment and creates default communicator. Must be called before any other MPI routine

  (let* ((config-file (options 'config-file) ) ; this let defines the config variables and the mpi communicator worlds
         (config-alst (or (and config-file (load-config-file config-file)) '())) ; load config file or give back empty list.
         (config      (lambda (k) (alist-ref k config-alst))) 
	       (comm-world (MPI:get-comm-world)) ; returns the default communicator that was created by MPI_Init. Group associated to it contains all processes
	       (myrank     (MPI:comm-rank comm-world)) ; return rank of the calling process of the common world
	       (my-comm    (let ((color (options 'mpi-split)))
		       (if color 
			     (MPI:comm-split comm-world color myrank) ; create new communicators 
           ;based on a color (an int (?)), that assigns to which communicator the new process will belong (all processes with the same color belong to same comm.)
           ;the second value (key) assigns the rank of the process in the new comm.
			     comm-world)))
	       (size (MPI:comm-size my-comm)) ; retunrns size of group that is associated with my_comm
	      )

    (if (zero? myrank) ; if this is the master process (rank = 0), print what is in the config file.
        (begin
          (print "Brep config = " )
          (pp config-alst))) ; prints what is in the configuration file.

    (let* (
	    (nGC (or (config 'numGC) (options 'num-gc) (defopt 'num-gc)))
      (pf-z-offset (or 
          (and (config 'PCLdepth) (config 'GLdepth) (+ (config 'PCLdepth) (config 'GLdepth)) )
          (options 'aa-length) 
          (defopt 'aa-length)))
	    (x-extent (exact->inexact (or (config 'GoCxrange) (options 'x-extent) (defopt 'x-extent))))
	    (y-extent (exact->inexact (or (config 'GoCyrange) (options 'y-extent) (defopt 'y-extent))))
      (z-extent (exact->inexact (or (config 'GoCzrange) (options 'z-extent) (defopt 'z-extent))))
      (boundary (XZAxis 3 1 
          (f64vector 0. (/ x-extent 2.) x-extent)
          (f64vector z-extent z-extent z-extent)
          (Bounds (make-bounds y-extent 0. 0. x-extent))))

 	    (GoC-Soma-Points 
	        (if (options 'goc-points)
		          (car (load-points-from-file (options 'goc-points)))
              (let ((GoC-grid (car (Grid (/ x-extent 2.) (/ y-extent 2.) (/ z-extent 2.) boundary))))
                  (car (ClusteredRandomPointProcess 
                      GoC-grid
                      (or (config 'numGoC) (options 'num-goc) (defopt 'num-goc))
                      (or (config 'meanGoCdistance) (options 'mean-goc-distance) (defopt 'mean-goc-distance)) 
                      (car (get-rng-seeds)) (car (get-rng-seeds) )
                      boundary)))))

	   (GoCs (let ((dendrite-labels   '(BasolateralDendrites ApicalDendrites ))
                       (Anseg       (or (config 'GoC_Ad_nseg) (options 'goc-apical-nseg) (defopt 'goc-apical-nseg)))
                       (Bnseg       (or (config 'GoC_Bd_nseg) (options 'goc-basolateral-nseg) (defopt 'goc-basolateral-nseg)))
                       (Ansegpts    (or (config 'GoC_Ad_nsegpts) (options 'goc-apical-nsegpts) (defopt 'goc-apical-nsegpts)))
                       (Bnsegpts    (or (config 'GoC_Bd_nsegpts) (options 'goc-basolateral-nsegpts) (defopt 'goc-basolateral-nsegpts)))
                       (Adendh      (or (config 'GoC_PhysApicalDendH) (options 'goc-apical-dendheight) (defopt 'goc-apical-dendheight)))
                       (Bdendh      (or (config 'GoC_PhysBasolateralDendH) (options 'goc-basolateral-dendheight) (defopt 'goc-basolateral-dendheight)))
                       (Aradius     (or (config 'GoC_PhysApicalDendR) (options 'goc-apical-radius) (defopt 'goc-apical-radius)))
                       (Bradius     (or (config 'GoC_PhysBasolateralDendR) (options 'goc-basolateral-radius) (defopt 'goc-basolateral-radius)))
                       (ndendrites  (or (config 'numDendGolgi) (options 'goc-dendrites) (defopt 'goc-dendrites)))
                       (Atheta-range
                        (cons (or (config 'GoC_Atheta_min) (options 'goc-theta-apical-min) (defopt 'goc-theta-apical-min))
                              (or (config 'GoC_Atheta_max) (options 'goc-theta-apical-max) (defopt 'goc-theta-apical-max))))
                       (Btheta-range
                        (cons (or (config 'GoC_Btheta_min) (options 'goc-theta-basolateral-min) (defopt 'goc-theta-basolateral-min))
                              (or (config 'GoC_Btheta_max) (options 'goc-theta-basolateral-max) (defopt 'goc-theta-basolateral-max))))
                       (Atheta-stdev
                        (or (config 'GoC_Atheta_stdev) (options 'goc-theta-apical-stdev) (defopt 'goc-theta-apical-stdev)))
                       (Btheta-range
                        (cons (or (config 'GoC_Btheta_min) (options 'goc-theta-basolateral-min) (defopt 'goc-theta-basolateral-min))
                              (or (config 'GoC_Btheta_max) (options 'goc-theta-basolateral-max) (defopt 'goc-theta-basolateral-max))))
                       (Btheta-stdev
                        (or (config 'GoC_Btheta_stdev) (options 'goc-theta-basolateral-stdev) (defopt 'goc-theta-basolateral-stdev)))
                       (axon-x-range
                        (cons (or (config 'GoC_Axon_Xmin) (options 'goc-axon-x-min) (defopt 'goc-axon-x-min))
                              (or (config 'GoC_Axon_Xmax) (options 'goc-axon-x-max) (defopt 'goc-axon-x-max))))
                       (axon-y-range
                        (cons (or (config 'GoC_Axon_Ymin) (options 'goc-axon-y-min) (defopt 'goc-axon-y-min))
                              (or (config 'GoC_Axon_Ymax) (options 'goc-axon-y-max) (defopt 'goc-axon-y-max))))
                       (axon-z-range
                        (cons (or (config 'GoC_Axon_Zmin) (options 'goc-axon-z-min) (defopt 'goc-axon-z-min))
                              (or (config 'GoC_Axon_Zmax) (options 'goc-axon-z-max) (defopt 'goc-axon-z-max))))

                       )

                   (let* (
                          (GoC-Soma-Points-lst (kd-tree->list GoC-Soma-Points))
                          
                          (dendrites
                           (ParametricNeurites 
                            dendrite-labels
                            (list (/ ndendrites 2) (/ ndendrites 2))
                            (list (inexact->exact Bnseg) (inexact->exact Anseg))
                            (list (inexact->exact Bnsegpts) (inexact->exact Ansegpts))
                            GoC-Soma-Points-lst
                            (car (get-rng-seeds))
                            ConePerturbationNeurites
                            (list (list Btheta-range Btheta-stdev Bdendh Bradius)
                                  (list Atheta-range Atheta-stdev Adendh Aradius))
                            ))

                          (axons
                           (ParametricNeurites 
                            '(Axons)
                            (list (or (config 'numAxonGolgi) (options 'goc-axons) (defopt 'goc-axons)))
                            (list (inexact->exact (or (config 'GoC_Axon_nseg) (options 'goc-axonsegs) (defopt 'goc-axonsegs))))
                            (list (inexact->exact (or (config 'GoC_Axon_npts) (options 'goc-axonpts) (defopt 'goc-axonpts))))
                            GoC-Soma-Points-lst
                            (car (get-rng-seeds))
                            LinearPerturbationNeurites
                            (list (list axon-x-range
                                        axon-y-range
                                        axon-z-range))
                            ))
                          )

                     (reverse
                      (car 
                       (fold
                        (lambda (p ds as cells.gi)
                          (let ((cells (car cells.gi))  (gi (cadr cells.gi)))
                            (list (cons (make-cell 'GoC gi p (append ds as)) cells) (+ 1 gi) )))
                        (list '() 0)
                        GoC-Soma-Points-lst dendrites axons)))

                     ))
                 )

	  (GC-Points 
      (if (options 'gc-points)
			  (car (load-points-from-file (options 'gc-points)))
			  (car (UniformRandomPointProcess 
                                nGC (car (get-rng-seeds)) (car (get-rng-seeds) )  boundary))))
    ; read in gct points, default is 
	  (GCT-Points 
      (if (options 'gct-points)
			  (car (load-points-from-file (options 'gct-points)))
        (kd-tree-map 
          (lambda (p) 
            (let ((x (coord 0 p))
                  (y (coord 1 p))
                  (z (+ pf-z-offset (coord 2 p))))
             (make-point x y z)))
          GC-Points)))

	   )

	  
      (if (zero? myrank)
          
          (begin

            (call-with-output-file "GoCcoordinates.sorted.dat"
              (lambda (out)
                (for-each (lambda (gx) 
                            (let ((x (cell-origin gx)))
                              (fprintf out "~A ~A ~A~%" 
                                       (coord 0 x)
                                       (coord 1 x)
                                       (coord 2 x))))
                          GoCs)))

            (call-with-output-file "GoCadendcoordinates.sorted.dat"
              (write-sections 'ApicalDendrites GoCs))
            
            (call-with-output-file "GoCbdendcoordinates.sorted.dat"
              (write-sections 'BasolateralDendrites GoCs))
            
            (call-with-output-file "GoCaxoncoordinates.sorted.dat"
              (write-sections 'Axons GoCs))
            
            (call-with-output-file "GCcoordinates.sorted.dat"
              (lambda (out)
                (for-each (lambda (x) 
                            (fprintf out "~A ~A ~A~%" 
                                     (coord 0 x)
                                     (coord 1 x)
                                     (coord 2 x)))
                          (kd-tree->list GC-Points))))
            
            (call-with-output-file "GCTcoordinates.sorted.dat"
              (lambda (out)
                (for-each (lambda (x) 
                            (fprintf out "~A ~A ~A~%" 
                                     (coord 0 x)
                                     (coord 1 x)
                                     (coord 2 x)))
                          (kd-tree->list GCT-Points))))
            ))

      (d "Golgi and PF point sets constructed~%")
      
      (gc)

      (MPI:barrier comm-world) ; synchronizes barrier
      
      (let* (
             (my-GC-Points
	      (let recur ((gcps (kd-tree->list* GC-Points)) (myindex 0) (ax '()))
		(if (null? gcps) ax
		    (let ((ax1 (if (= (modulo myindex size) myrank)
				   (cons (car gcps) ax) ax)))
		      (recur (cdr gcps) (+ 1 myindex) ax1)))))

             (my-GCT-Points
	      (let recur ((gcps (kd-tree->list* GCT-Points)) (myindex 0) (ax '()))
		(if (null? gcps) ax
		    (let ((ax1 (if (= (modulo myindex size) myrank)
				   (cons (car gcps) ax) ax)))
		      (recur (cdr gcps) (+ 1 myindex) ax1)))))

	     (my-AAs-tree
	      (sections->kd-tree
               (ParametricNeurites*
                '(AscendingAxon) (list 1)
                my-GC-Points 
                (car (get-rng-seeds))
                LinearNeurites
                (list
                 (list
                  (or (config 'AAstep) (options 'aa-step) (defopt 'aa-step))
                  (or (and (config 'PCLdepth) (config 'GLdepth)
                           (+ (config 'PCLdepth) (config 'GLdepth)) )
                      (options 'aa-length) (defopt 'aa-length))
                 2 (make-point 0. 0. 0.)))
                )
	       ))

             (_
              (if (options 'save-aa) ; boolean function, when true, the ascending axon points are saved.
                  (call-with-output-file (sprintf "AAcoordinates~A.dat" (if (> size 1) myrank ""))
                    (lambda (out)
                      (for-each (lambda (x) 
                                  (let ((p (cadr x))
                                        (i (car (cadar x))))
                                    (fprintf out "~A ~A ~A ~A~%" 
                                             i
                                             (coord 0 p)
                                             (coord 1 p)
                                             (coord 2 p))))
                                (kd-tree->list* my-AAs-tree))))
                  ))

	     (_ (d "rank ~A: ascending axons constructed~%" myrank))
	     
	     (my-PFs-tree
	      (sections->kd-tree
               (ParametricNeurites*
                '(ParallelFiber) (list 1)
                my-GCT-Points 
                (car (get-rng-seeds))
                LinearNeurites
                (list
                 (list (or (config 'PFstep) (options 'pf-step) (defopt 'pf-step))
                       (or (config 'PFlength) (options 'pf-length) (defopt 'pf-length))
                       0 (make-point (- (/ (or (config 'PFlength) (options 'pf-length) (defopt 'pf-length)) 2)) 0. 0.)))
                 ))
	       )

             
             (_
              (if (options 'save-pf)
                  (call-with-output-file (sprintf "PFcoordinates~A.dat" (if (> size 1) myrank ""))
                    (lambda (out)
                      (for-each (lambda (x) 
                                  (let ((i (car (cadar x)))
                                        (p (cadr x)))
                                  (fprintf out "~A ~A ~A ~A~%" 
                                           i
                                           (coord 0 p)
                                           (coord 1 p)
                                           (coord 2 p))))
                                (kd-tree->list* my-PFs-tree))))
                  ))

	     (_ (d "rank ~A: parallel fibers constructed~%" myrank))

	     (GoC-ApicalDendrites (map (lambda (gx) (list (cell-index gx) (cell-section-ref 'ApicalDendrites gx))) GoCs))
	     (GoC-BasolateralDendrites (map (lambda (gx) (list (cell-index gx) (cell-section-ref 'BasolateralDendrites gx))) GoCs))
	     (GoC-Axons (map (lambda (gx) (list (cell-index gx) (cell-section-ref 'Axons gx))) GoCs))

	     (my-GoCs
	      (let recur ((cells GoCs) (myindex 0) (ax '()))
		(if (null? cells) ax
		    (let ((ax1 (if (= (modulo myindex size) myrank)
				   (cons (car cells) ax) ax)))
		      (recur (cdr cells) (+ 1 myindex) ax1)))
		))

	     (_ (d "rank ~A: length my-GoCs = ~A ~%" myrank (length my-GoCs)))

	     (my-GoC-Soma-tree (cells-origins->kd-tree my-GoCs))

	     (my-GoC-Axons-tree (cells-sections->kd-tree my-GoCs 'Axons))

	     (_ (d "rank ~A: GoC axon tree constructed~%" myrank))
	     
	     )
	
	(gc)

	(GC-GoC-connections (list GoC-ApicalDendrites) my-PFs-tree my-comm myrank size
                            (or (options 'output) "PFtoGoC")
                            (or (config 'PFtoGoCzone) (options 'pf-goc-zone) (defopt 'pf-goc-zone))
                            (or (options 'pf-start) 0)
                            (or (options 'goc-start) 0)
                            )

	(MPI:barrier comm-world)

	(GC-GoC-connections (list GoC-ApicalDendrites GoC-BasolateralDendrites )
                            my-AAs-tree my-comm myrank size
                            (or (options 'output) "AAtoGoC")
                            (or (config 'AAtoGoCzone) (options 'aa-goc-zone) (defopt 'aa-goc-zone))
                            (or (options 'pf-start) 0)
                            (or (options 'goc-start) 0)
                            )

	(MPI:barrier comm-world)

        (if (zero? myrank)
            (GoC-GoC-distances (kd-tree->list* GoC-Soma-Points) my-comm myrank size
                               (or (options 'output) "GoC")
                               (or (options 'goc-start) 0)))

	(GoC-GoC-connections (kd-tree->list* GoC-Soma-Points) my-GoC-Axons-tree my-comm myrank size 
                             (or (options 'output) "GoCtoGoC")
                             (or (config 'GoCtoGoCzone) (options 'goc-goc-zone) (defopt 'goc-goc-zone))
                             (or (options 'goc-start) 0)
			     nn-filter: (lambda (x nns)
					  (let ((xz (coord 2 x)))
					    (filter (lambda (nn) (let ((nz (coord 2 (cadr nn)))) 
                                                       (positive? (- nz xz)))) nns))))

	(MPI:barrier comm-world)

	(GoC-GoC-gap-connections (kd-tree->list* GoC-Soma-Points) my-GoC-Soma-tree my-comm myrank size
                                 (or (options 'output) "GoCtoGoCgap")
                                 (or (config 'GoCtoGoCgapzone) (options 'goc-goc-gap-zone) (defopt 'goc-goc-gap-zone))
                                 (or (options 'goc-start) 0))
	
	(MPI:finalize) ; terminates this environment
	
	))
))

(width 30)
(main opt (opt '@))

;  the '@ key comes from the getopt-long grammar and specifies a list of arguments that are not options or option values

