import alignfaces as afa

faces_path = "./face/"
afa.get_landmarks(faces_path)
aligned_path = afa.align_procrustes(faces_path)
afa.get_landmarks(aligned_path)
the_aperture, aperture_path = afa.place_aperture(aligned_path)
