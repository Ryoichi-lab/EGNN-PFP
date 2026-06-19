#!/bin/bash
#PBS -N orca_pbe0_t
#PBS -q workq
#PBS -l select=1:ncpus=16:mpiprocs=16:mem=150GB
#PBS -l walltime=24:00:00
#PBS -j oe

LOCATION=/home/users/uchiyama/relativistic_effect/input_orca_pbe0_descriptor_rich_test_rerun1
suffixInput="inp"
suffixOutput="out"
doCompress=y
overall_status=0

echo "----- ORCA PBE0 descriptor-rich test batch -----"

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

nNode=$(cat "${PBS_NODEFILE}" | uniq | wc -l)
nProcess=$(cat "${PBS_NODEFILE}" | wc -l)

NodeList=()
for i in $(seq 1 "${nNode}"); do
    node=$(cat "${PBS_NODEFILE}" | uniq | head -n "${i}" | tail -n 1 | awk '{print $1}' | sed -s "s/\\.seinolab\\.ac\\.jp//g")
    NodeList=( "${NodeList[@]}" "${node}" )
done

echo "Nodes: ${NodeList[*]}"
echo "Processes: ${nProcess}"
echo "Location: ${LOCATION}"

WRK0="/work/$USER/${PBS_JOBID}"
for node in "${NodeList[@]}"; do
    ssh "${node}" mkdir -p "${WRK0}" > /dev/null
done

module load openMPI/4.1.6
export OMPI_MCA_btl_tcp_if_include=eth0
export OMPI_MCA_plm_rsh_disable_qrsh=true
export OMPI_MCA_plm_rsh_agent=ssh

export PKGROOT=/home/share/packages/orca/6.0.0/bin
export PATH="${PKGROOT}:${PATH}"

FILENAMES=( $(cd "${LOCATION}" && ls *.${suffixInput} 2>/dev/null | sed "s/\.${suffixInput}$//") )

for JOB in "${FILENAMES[@]}"; do
    INPUT_ORIGINAL="${LOCATION}/${JOB}.${suffixInput}"
    OUTPUT="${LOCATION}/${JOB}.${suffixOutput}"
    DATADIR="${LOCATION}/${JOB}"
    WRK="${WRK0}/${JOB}"

    if [ -e "${OUTPUT}.gz" ]; then
        echo "SKIP (done): ${JOB}"
        continue
    fi

    mkdir -p "${WRK}" > /dev/null
    for node in "${NodeList[@]}"; do
        ssh "${node}" mkdir -p "${WRK}" > /dev/null
        rsync -ahrc "${INPUT_ORIGINAL}" "${node}:${WRK}/${JOB}.${suffixInput}"
    done

    cd "${WRK}" || exit 1
    echo "Running ${JOB}"
    "${PKGROOT}/orca" "${JOB}.${suffixInput}" >> "${OUTPUT}" 2>&1
    ORCA_EXIT=$?
    if [ ${ORCA_EXIT} -ne 0 ]; then
        echo "ORCA failed for ${JOB} with exit code ${ORCA_EXIT}" >> "${OUTPUT}"
        overall_status=${ORCA_EXIT}
    fi
    cd "${LOCATION}" || exit 1

    for node in "${NodeList[@]}"; do
        mkdir -p "${DATADIR}/${node}"
        rsync -ahrc "${node}:${WRK}/*.gbw" "${DATADIR}/${node}/" 2> /dev/null
        rsync -ahrc "${node}:${WRK}/*.xyz" "${DATADIR}/${node}/" 2> /dev/null
        rsync -ahrc "${node}:${WRK}/*.engrad" "${DATADIR}/${node}/" 2> /dev/null
        rsync -ahrc "${node}:${WRK}/*.property.txt" "${DATADIR}/${node}/" 2> /dev/null
        rsync -ahrc "${node}:${WRK}/*.densities" "${DATADIR}/${node}/" 2> /dev/null
        rsync -ahrc "${node}:${WRK}/*.scfp" "${DATADIR}/${node}/" 2> /dev/null
        rsync -ahrc "${node}:${WRK}/*.scfr" "${DATADIR}/${node}/" 2> /dev/null
    done

    if [ "${doCompress,,}" == "y" ] || [ "${doCompress,,}" == "yes" ] || [ -e "${OUTPUT}" ]; then
        gzip -f "${OUTPUT}"
    fi

    for node in "${NodeList[@]}"; do
        ssh "${node}" rm -fr "${WRK}" > /dev/null
    done
done

for node in "${NodeList[@]}"; do
    ssh "${node}" rm -fr "${WRK0}" > /dev/null
done

echo "All done: $(date)"
exit ${overall_status}
