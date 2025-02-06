module cam_nn
#ifdef USE_FTORCH
  use ftorch,         only: torch_kCUDA, torch_tensor, torch_model, torch_tensor_from_array, torch_kcpu
  use ftorch,         only: torch_model_load, torch_model_forward, torch_tensor_delete
  use camsrfexch,     only: cam_in_t
  use physics_types,  only: physics_state
  use shr_kind_mod,   only: CL=>shr_kind_cl
  use spmd_utils,     only: masterproc, mpicom, mpi_character
  use cam_abortutils,only: endrun
  implicit none

  public :: torch_inference, torch_readnl

  ! Declare the torch model without initializing it here
  type(torch_model) :: model
  logical           :: model_initialized = .false.
  character(len=CL) :: weights_file

contains

  subroutine torch_readnl(nlfile)
    use namelist_utils, only : find_group_name
    character(len=*), intent(in) :: nlfile
    integer :: unitn
    integer :: ierr
    character(len=*), parameter :: sub="torch_readnl"

    namelist /torch_nl/ weights_file

    if(masterproc) then
       open( newunit=unitn, file=trim(nlfile), status='old' )
       call find_group_name(unitn, 'torch_nl', status=ierr)
       if (ierr == 0) then
          read(unitn, torch_nl, iostat=ierr)
          if (ierr /= 0) then
             call endrun(sub//': FATAL: reading namelist')
          end if
       end if
       close(unitn)
    endif
   ! Broadcast namelist variables
   call mpi_bcast(weights_file,         CL, mpi_character, 0, mpicom, ierr)
   if (ierr /= 0) call endrun("torch_readnl: FATAL: mpi_bcast: weights_file")

 end subroutine torch_readnl

  subroutine init_torch_model(model)
    ! Initialize the model
    type(torch_model), intent(inout) :: model
    character(len=*), parameter :: sub="init_torch_model"

    call torch_model_load(model, weights_file, torch_kCUDA)
  end subroutine init_torch_model

  subroutine torch_inference(phys_state)
    ! CAM Types
    type(physics_state), intent(inout) :: phys_state(:)

    ! Torch Types
    type(torch_tensor), dimension(3) :: in_tensors
    type(torch_tensor), dimension(2) :: out_tensors

    ! Make input tensors
    real(8), allocatable :: phys_state_t_array(:,:,:)
    real(8), allocatable :: phys_state_pmid_array(:,:,:)
    real(8), allocatable :: phys_state_q_array(:,:,:,:)
    real(8), allocatable :: new_phys_state_t_array(:,:,:)
    real(8), allocatable :: new_phys_state_q_array(:,:,:,:)

    integer :: tensor_layout_3d(3) = [3,2,1]
    integer :: tensor_layout_4d(4) = [4,3,2,1]

    ! Integers
    integer :: i, m, n

    ! Initialize the model if it has not been initialized yet
    if (.not. model_initialized) then
       call init_torch_model(model)
       model_initialized = .true.
    end if

    ! Make Temp/pressure Tensor
    m = size(phys_state(1)%t, 1)  ! Number of columns/cells
    n = size(phys_state(1)%t, 2)  ! Number of levels

    allocate(phys_state_t_array(size(phys_state), m, n))
    allocate(phys_state_pmid_array(size(phys_state), m, n))
    allocate(new_phys_state_t_array(size(phys_state), m, n))

    ! Make Mixing ratio Tensor
    m = size(phys_state(1)%q, 1)  ! Number of columns/cells
    n = size(phys_state(1)%q, 2)  ! Number of levels
    i = size(phys_state(1)%q, 3)  ! Number of species (for mixing ratio)

    allocate(phys_state_q_array(size(phys_state), m, n, i))
    allocate(new_phys_state_q_array(size(phys_state), m, n, i))

    ! Fill our local arrays from the phys_state
    do i = 1, size(phys_state)
       phys_state_t_array(i, :, :)    = phys_state(i)%t
       phys_state_pmid_array(i, :, :) = phys_state(i)%pmid
       phys_state_q_array(i, :, :, :) = phys_state(i)%q
    end do

    ! Make Torch Tensors for input and output
    call torch_tensor_from_array(in_tensors(1), phys_state_t_array,    tensor_layout_3d, torch_kCUDA)
    call torch_tensor_from_array(in_tensors(2), phys_state_pmid_array, tensor_layout_3d, torch_kCUDA)
    call torch_tensor_from_array(in_tensors(3), phys_state_q_array,    tensor_layout_4d, torch_kCUDA)

    call torch_tensor_from_array(out_tensors(1), new_phys_state_t_array, tensor_layout_3d, torch_kCPU)
    call torch_tensor_from_array(out_tensors(2), new_phys_state_q_array, tensor_layout_4d, torch_kCPU)

    ! Perform inference
    call torch_model_forward(model, in_tensors, out_tensors)

    ! Copy the output tensor back to the physics state
    do i = 1, size(phys_state)
       phys_state(i)%t = new_phys_state_t_array(i, :, :)
       phys_state(i)%q = new_phys_state_q_array(i, :, :, :)
    end do

    ! Free the torch tensors
    call torch_tensor_delete(in_tensors(1))
    call torch_tensor_delete(in_tensors(2))
    call torch_tensor_delete(in_tensors(3))
    call torch_tensor_delete(out_tensors(1))
    call torch_tensor_delete(out_tensors(2))

    ! Deallocate the local arrays
    if (allocated(phys_state_t_array))    deallocate(phys_state_t_array)
    if (allocated(phys_state_pmid_array)) deallocate(phys_state_pmid_array)
    if (allocated(phys_state_q_array))    deallocate(phys_state_q_array)
    if (allocated(new_phys_state_t_array)) deallocate(new_phys_state_t_array)
    if (allocated(new_phys_state_q_array)) deallocate(new_phys_state_q_array)

  end subroutine torch_inference
#endif
end module cam_nn
