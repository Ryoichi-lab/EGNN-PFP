#!/bin/bash
#PBS -N nmr_nr_b
#PBS -q workq
#PBS -l select=1:ncpus=4:mpiprocs=4:mem=40GB:host=green15
#PBS -l walltime=10000:00:00
#PBS -j oe

echo "----- ORCA non-rel NMR: batch_b (green15, 4CPU) -----"
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

module load openMPI/4.1.6
export OMPI_MCA_btl_tcp_if_include=eth0
export OMPI_MCA_plm_rsh_disable_qrsh=true
export OMPI_MCA_plm_rsh_agent=ssh

export PKGROOT=/home/share/packages/orca/6.0.0/bin
LOCATION=/home/users/uchiyama/relativistic_effect/input_orca_nmr_nonrel/batch_b
LIST_FILE=/home/users/uchiyama/relativistic_effect/input_orca_nmr_nonrel/lists/batch_b.list

WRK="/work/$USER/${PBS_JOBID}"
mkdir -p "${WRK}"

DONE=0; SKIP=0; FAIL=0
echo "Node: $(hostname),  Start: $(date)"
echo "Total: $(wc -l < ${LIST_FILE}) jobs"

mapfile -t JOBS < "${LIST_FILE}"

for JOB in "${JOBS[@]}"; do
    [[ -z "${JOB}" ]] && continue
    INPUT="${LOCATION}/${JOB}.inp"
    OUTPUT="${LOCATION}/${JOB}.out"

    [[ ! -f "${INPUT}" ]] && { SKIP=$((SKIP+1)); continue; }
    if [[ -f "${OUTPUT}.gz" ]]; then echo "SKIP: ${JOB}"; SKIP=$((SKIP+1)); continue; fi

    WRKJOB="${WRK}/${JOB}"
    mkdir -p "${WRKJOB}"
    cp "${INPUT}" "${WRKJOB}/${JOB}.inp"
    cd "${WRKJOB}"

    "${PKGROOT}/orca" "${JOB}.inp" > "${OUTPUT}" 2>&1
    EXIT=$?

    if grep -q 'ORCA TERMINATED NORMALLY' "${OUTPUT}" 2>/dev/null; then
        gzip -f "${OUTPUT}"
        DONE=$((DONE+1))
    else
        mv "${OUTPUT}" "${OUTPUT}.failed" 2>/dev/null
        FAIL=$((FAIL+1))
        echo "FAIL: ${JOB} (exit=${EXIT})"
    fi

    rm -rf "${WRKJOB}"
    cd "${LOCATION}"
done

rm -rf "${WRK}"
echo "Done: ${DONE}  Skip: ${SKIP}  Fail: ${FAIL}  $(date)"
